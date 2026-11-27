"""How well a pending queue job fits the filament a printer has loaded right now.

The scheduler answers this question every pass, but only on the way to a
dispatch decision, and only for the one printer it is about to use. The queue
page, the printer card and planning views ask it for every pending job against
every printer the job could go to, without dispatching anything. Until now the
frontend answered it with its own copy of the matcher
(``filterCompatibleQueueItems`` in ``frontend/src/utils/printer.ts``), which has
to be kept in step with the backend by hand.

This module answers it read-only, by asking the scheduler's own helpers
(``_compute_ams_mapping_for_printer``, ``_unmatched_requirements``,
``_short_filaments_on``, ``_find_color_substitutions``), so its answer is the
one dispatch would reach on the same printer state. Nothing here writes to the
queue row: a variant job is evaluated on a transient copy, never on the row.

The mapping is computed fresh from what is loaded. A pinned job's stored
``ams_mapping`` is deliberately not consulted: the question is "would this job
go out on what is loaded now", and a stored mapping describes the trays as
they were when the job was queued.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.print_queue import PrintQueueItem, PrintQueueVariant
from backend.app.services.queue_todo import TodoFilament, todo_filament_from_requirement
from backend.app.utils.printer_models import normalize_printer_model

if TYPE_CHECKING:
    from backend.app.services.print_scheduler import PrintScheduler

logger = logging.getLogger(__name__)

FitStatus = Literal["fits", "unmatched", "short", "unknown"]

# Gcode states with no print loaded, as the frontend's IDLE_PRINTER_STATES.
_IDLE_STATES = frozenset({"IDLE", "FINISH", "FAILED"})


@dataclass(frozen=True)
class DemandLine:
    """One filament slot a job prints, after the job's own overrides.

    ``grams`` is what the plate prints of it; 0.0 when the file does not say.
    """

    slot_id: int
    type: str
    color: str
    color_name: str
    tray_info_idx: str
    grams: float
    force_color_match: bool
    nozzle_id: int | None = None


@dataclass(frozen=True)
class PrinterFit:
    """How one job fits one printer's loaded filament.

    ``status``:

    * ``fits`` -- every slot the plate prints has a tray, and no spool is too light.
    * ``unmatched`` -- at least one slot has no tray; ``unmatched`` names them.
    * ``short`` -- every slot has a tray, but a spool holds too little; ``short`` names them.
    * ``unknown`` -- nothing to judge by: no status for the printer, or a file
      with no readable filament requirements.

    ``swaps_needed`` counts the spools someone would have to load or replace.
    ``color_substitutions`` lists the slots that would print in another colour
    of the same material, in the shape the queue row stores (#3279).
    """

    printer_id: int
    status: FitStatus
    online: bool = False
    idle: bool = False
    unmatched: tuple[TodoFilament, ...] = ()
    short: tuple[TodoFilament, ...] = ()
    color_substitutions: tuple[dict, ...] = field(default=())

    @property
    def swaps_needed(self) -> int:
        return len(self.unmatched) + len(self.short)


def _overrides_by_slot(item: PrintQueueItem) -> dict[int, dict]:
    try:
        overrides = json.loads(item.filament_overrides) if item.filament_overrides else []
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(overrides, list):
        return {}
    return {o["slot_id"]: o for o in overrides if isinstance(o, dict) and isinstance(o.get("slot_id"), int)}


async def job_demand(db: AsyncSession, item: PrintQueueItem, scheduler: PrintScheduler) -> list[DemandLine]:
    """The filament slots ``item`` prints, with its overrides applied.

    Read from the same parse the scheduler matches against
    (``_get_filament_requirements`` then ``_apply_filament_overrides``), so the
    type and colour are the ones dispatch will look for. Empty when the file
    has no readable requirements.
    """
    required = await scheduler._get_filament_requirements(db, item)
    if not required:
        return []
    scheduler._apply_filament_overrides(item, required)
    overrides = _overrides_by_slot(item)
    lines = []
    for req in required:
        slot_id = req.get("slot_id")
        if not isinstance(slot_id, int):
            continue
        try:
            grams = float(req.get("used_grams") or 0.0)
        except (TypeError, ValueError):
            grams = 0.0
        nozzle = req.get("nozzle_id")
        lines.append(
            DemandLine(
                slot_id=slot_id,
                type=str(req.get("type") or "").upper(),
                color=str(req.get("color") or ""),
                color_name=str(req.get("color_name") or ""),
                tray_info_idx=str(req.get("tray_info_idx") or ""),
                grams=grams,
                force_color_match=bool(overrides.get(slot_id, {}).get("force_color_match")),
                nozzle_id=nozzle if isinstance(nozzle, int) else None,
            )
        )
    return lines


def variant_for_model(item: PrintQueueItem, printer_model: str | None) -> PrintQueueVariant | None:
    """The variant ``item`` would print on a printer of ``printer_model``, best first (#671).

    None when the item has no variants or none is for that model.
    """
    if not item.variants or not printer_model:
        return None
    wanted = (normalize_printer_model(printer_model) or printer_model).lower()
    for variant in sorted(item.variants, key=lambda v: (v.position, v.id or 0)):
        if (normalize_printer_model(variant.target_model) or variant.target_model).lower() == wanted:
            return variant
    return None


def as_variant_item(item: PrintQueueItem, variant: PrintQueueVariant) -> PrintQueueItem:
    """A transient copy of ``item`` carrying ``variant``'s file and settings.

    The scheduler folds the winning variant onto the row itself
    (``_resolve_variant``); a read-only check must not, so it works on a copy
    that is never added to a session.
    """
    return PrintQueueItem(
        id=item.id,
        printer_id=item.printer_id,
        target_model=variant.target_model,
        target_location=item.target_location,
        library_file_id=variant.library_file_id,
        library_file=variant.library_file,
        archive_id=None,
        plate_id=variant.plate_id,
        filament_overrides=variant.filament_overrides,
        required_filament_types=variant.required_filament_types,
        skip_filament_check=item.skip_filament_check,
        created_by_id=item.created_by_id,
    )


async def fit_on_printer(
    db: AsyncSession,
    item: PrintQueueItem,
    printer_id: int,
    scheduler: PrintScheduler,
    printer_model: str | None = None,
) -> PrinterFit:
    """How ``item`` fits what ``printer_id`` has loaded now. Changes nothing.

    ``printer_model`` picks the variant of a cross-model job; it is ignored
    for an ordinary job. ``item.variants`` must be loaded already (the async
    session cannot lazy-load it), as the scheduler's own queue query does. Any failure reads as ``unknown`` rather than raising:
    a planning view must not break on one unreadable file.
    """
    from backend.app.services.printer_manager import printer_manager

    status = printer_manager.get_status(printer_id)
    if status is None:
        return PrinterFit(printer_id=printer_id, status="unknown")
    online = bool(printer_manager.is_connected(printer_id))
    idle = online and (status.state or "").upper() in _IDLE_STATES

    if item.variants:
        variant = variant_for_model(item, printer_model)
        if variant is None:
            return PrinterFit(printer_id=printer_id, status="unknown", online=online, idle=idle)
        item = as_variant_item(item, variant)

    try:
        required = await scheduler._get_filament_requirements(db, item)
        if not required:
            return PrinterFit(printer_id=printer_id, status="unknown", online=online, idle=idle)
        scheduler._apply_filament_overrides(item, required)

        mapping = await scheduler._compute_ams_mapping_for_printer(db, printer_id, item)
        unmatched_reqs = await scheduler._unmatched_requirements(db, item, printer_id, mapping)
        if unmatched_reqs:
            return PrinterFit(
                printer_id=printer_id,
                status="unmatched",
                online=online,
                idle=idle,
                unmatched=tuple(todo_filament_from_requirement(r) for r in unmatched_reqs),
            )
        if mapping is None:
            # Nothing resolved and nothing reported loaded: no reading to judge by.
            return PrinterFit(printer_id=printer_id, status="unknown", online=online, idle=idle)

        short = await scheduler._short_filaments_on(db, item, printer_id)
        substitutions = scheduler._find_color_substitutions(
            required, mapping, scheduler._build_loaded_filaments(status)
        )
    except Exception as e:
        logger.debug("Queue item %s: filament fit on printer %s not read: %s", item.id, printer_id, e)
        return PrinterFit(printer_id=printer_id, status="unknown", online=online, idle=idle)

    return PrinterFit(
        printer_id=printer_id,
        status="short" if short else "fits",
        online=online,
        idle=idle,
        short=tuple(short),
        color_substitutions=tuple(substitutions),
    )
