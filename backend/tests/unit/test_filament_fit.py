"""How a pending job fits a printer's loaded filament, read-only (services/filament_fit.py).

The fit service must reach the verdict dispatch would reach on the same
printer state, without touching the queue row. These tests drive the real
matcher with fixture tray readings and only stand in for the 3MF parse and,
where a test is about the amount on the spool, the deficit check.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.models.library import LibraryFile
from backend.app.models.print_queue import PrintQueueItem, PrintQueueVariant
from backend.app.services.filament_fit import as_variant_item, fit_on_printer, job_demand, variant_for_model
from backend.app.services.print_scheduler import PrintScheduler
from backend.app.services.queue_todo import TodoFilament
from backend.tests.unit.test_scheduler_model_filament_short_3137 import (
    _add_item,
    _get_item,
    queue_db,  # noqa: F401 - fixture
)

WHITE_PLA = {"slot_id": 1, "type": "PLA", "color": "#FFFFFF", "tray_info_idx": "", "used_grams": 42.5}


def _tray(tray_id: int, tray_type: str, color: str) -> dict:
    return {"id": str(tray_id), "tray_type": tray_type, "tray_color": color, "tray_info_idx": "", "remain": 80}


def _status(*trays: dict, state: str = "IDLE") -> SimpleNamespace:
    return SimpleNamespace(
        raw_data={"ams": [{"id": "0", "tray": list(trays)}], "vt_tray": [], "ams_extruder_map": {}},
        nozzles=[],
        fila_switch=None,
        ams_filament_backup=None,
        state=state,
    )


def _patched(stack: ExitStack, status, *, connected: bool = True) -> None:
    pm = "backend.app.services.printer_manager.printer_manager"
    stack.enter_context(patch(f"{pm}.get_status", MagicMock(return_value=status)))
    stack.enter_context(patch(f"{pm}.is_connected", MagicMock(return_value=connected)))
    stack.enter_context(
        patch("backend.app.services.print_scheduler.printer_manager.get_status", MagicMock(return_value=status))
    )


async def _fit(ctx, item_id: int, status, requirements, *, short=None, printer_model="X1C"):
    scheduler = PrintScheduler()
    with ExitStack() as stack:
        _patched(stack, status)
        stack.enter_context(
            patch.object(
                scheduler,
                "_get_filament_requirements",
                AsyncMock(side_effect=lambda db, item: [dict(r) for r in requirements] or None),
            )
        )
        stack.enter_context(patch.object(scheduler, "_short_filaments_on", AsyncMock(return_value=short or [])))
        async with ctx.session_maker() as db:
            item = await _load(db, item_id)
            return await fit_on_printer(db, item, 1, scheduler, printer_model=printer_model)


async def _load(db, item_id: int) -> PrintQueueItem:
    """The row as the route loads it: variants eager, since async cannot lazy-load."""
    return (
        await db.execute(
            select(PrintQueueItem)
            .where(PrintQueueItem.id == item_id)
            .options(selectinload(PrintQueueItem.variants).selectinload(PrintQueueVariant.library_file))
        )
    ).scalar_one()


class TestVerdicts:
    @pytest.mark.asyncio
    async def test_same_filament_fits(self, queue_db):
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "FFFFFFFF")), [WHITE_PLA])
        assert fit.status == "fits"
        assert fit.swaps_needed == 0
        assert fit.color_substitutions == ()
        assert (fit.online, fit.idle) == (True, True)

    @pytest.mark.asyncio
    async def test_other_colour_of_the_same_material_fits_and_says_so(self, queue_db):
        """Without Force color match the matcher takes any spool of the type (#3279)."""
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "000000FF")), [WHITE_PLA])
        assert fit.status == "fits"
        assert [(s["slot_id"], s["wanted"], s["got"]) for s in fit.color_substitutions] == [(1, "#FFFFFF", "#000000")]

    @pytest.mark.asyncio
    async def test_other_material_is_unmatched(self, queue_db):
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PETG", "FFFFFFFF")), [WHITE_PLA])
        assert fit.status == "unmatched"
        assert fit.unmatched == (TodoFilament("PLA", "#FFFFFF"),)
        assert fit.swaps_needed == 1

    @pytest.mark.asyncio
    async def test_forced_colour_not_loaded_is_unmatched(self, queue_db):
        overrides = [{"slot_id": 1, "type": "PLA", "color": "#FFFFFF", "force_color_match": True}]
        item_id = await _add_item(queue_db, filament_overrides=json.dumps(overrides))
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "000000FF")), [WHITE_PLA])
        assert fit.status == "unmatched"
        assert [f.color for f in fit.unmatched] == ["#FFFFFF"]

    @pytest.mark.asyncio
    async def test_a_spool_too_light_is_short(self, queue_db):
        item_id = await _add_item(queue_db)
        light = TodoFilament("PLA", "#FFFFFF", short=True, required_grams=42.5)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "FFFFFFFF")), [WHITE_PLA], short=[light])
        assert fit.status == "short"
        assert fit.short == (light,)
        assert fit.swaps_needed == 1

    @pytest.mark.asyncio
    async def test_a_busy_printer_still_gets_a_verdict(self, queue_db):
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "FFFFFFFF"), state="RUNNING"), [WHITE_PLA])
        assert fit.status == "fits"
        assert fit.idle is False

    @pytest.mark.asyncio
    async def test_no_status_is_unknown(self, queue_db):
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, None, [WHITE_PLA])
        assert fit.status == "unknown"

    @pytest.mark.asyncio
    async def test_no_readable_requirements_is_unknown(self, queue_db):
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(_tray(0, "PLA", "FFFFFFFF")), [])
        assert fit.status == "unknown"

    @pytest.mark.asyncio
    async def test_nothing_loaded_is_unknown_not_unmatched(self, queue_db):
        """An empty reading is "never heard", not "nothing loaded" (#2589)."""
        item_id = await _add_item(queue_db)
        fit = await _fit(queue_db, item_id, _status(), [WHITE_PLA])
        assert fit.status == "unknown"

    @pytest.mark.asyncio
    async def test_a_failing_check_reads_unknown(self, queue_db):
        item_id = await _add_item(queue_db)
        scheduler = PrintScheduler()
        with ExitStack() as stack:
            _patched(stack, _status(_tray(0, "PLA", "FFFFFFFF")))
            stack.enter_context(
                patch.object(scheduler, "_get_filament_requirements", AsyncMock(side_effect=RuntimeError("boom")))
            )
            async with queue_db.session_maker() as db:
                item = await _load(db, item_id)
                fit = await fit_on_printer(db, item, 1, scheduler)
        assert fit.status == "unknown"


class TestAgreesWithDispatch:
    """The fit verdict matches the scheduler's own "would it be held here" answer."""

    CASES = [
        ([_tray(0, "PLA", "FFFFFFFF")], None),
        ([_tray(0, "PLA", "000000FF")], None),
        ([_tray(0, "PETG", "FFFFFFFF")], None),
        ([_tray(0, "PLA", "000000FF")], [{"slot_id": 1, "type": "PLA", "color": "#FFFFFF", "force_color_match": True}]),
        (
            [_tray(0, "PLA", "000000FF"), _tray(1, "PLA", "FFFFFFFF")],
            [{"slot_id": 1, "type": "PLA", "color": "#FFFFFF", "force_color_match": True}],
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("trays", "overrides"), CASES)
    async def test_fits_exactly_when_dispatch_would_not_hold(self, queue_db, trays, overrides):
        item_id = await _add_item(queue_db, filament_overrides=json.dumps(overrides) if overrides else None)
        status = _status(*trays)
        fit = await _fit(queue_db, item_id, status, [WHITE_PLA])

        scheduler = PrintScheduler()
        with ExitStack() as stack:
            _patched(stack, status)
            stack.enter_context(
                patch.object(
                    scheduler, "_get_filament_requirements", AsyncMock(side_effect=lambda db, i: [dict(WHITE_PLA)])
                )
            )
            async with queue_db.session_maker() as db:
                item = await _load(db, item_id)
                held_for = await scheduler._unmatched_on(db, item, 1)

        assert (fit.status == "fits") == (held_for == [])


class TestDemand:
    @pytest.mark.asyncio
    async def test_demand_applies_overrides_and_keeps_grams(self, queue_db):
        overrides = [
            {"slot_id": 1, "type": "PETG", "color": "#00FF00", "color_name": "Green", "force_color_match": True}
        ]
        item_id = await _add_item(queue_db, filament_overrides=json.dumps(overrides))
        scheduler = PrintScheduler()
        reqs = [WHITE_PLA, {"slot_id": 2, "type": "pla", "color": "#000000", "tray_info_idx": "GFA00", "used_grams": 3}]
        with patch.object(scheduler, "_get_filament_requirements", AsyncMock(return_value=[dict(r) for r in reqs])):
            async with queue_db.session_maker() as db:
                item = await _load(db, item_id)
                lines = await job_demand(db, item, scheduler)

        assert [(d.slot_id, d.type, d.color, d.color_name, d.grams, d.force_color_match) for d in lines] == [
            (1, "PETG", "#00FF00", "Green", 42.5, True),
            (2, "PLA", "#000000", "", 3.0, False),
        ]
        assert lines[1].tray_info_idx == "GFA00"

    @pytest.mark.asyncio
    async def test_no_requirements_is_no_demand(self, queue_db):
        item_id = await _add_item(queue_db)
        scheduler = PrintScheduler()
        with patch.object(scheduler, "_get_filament_requirements", AsyncMock(return_value=None)):
            async with queue_db.session_maker() as db:
                assert await job_demand(db, await _load(db, item_id), scheduler) == []


class TestVariants:
    async def _variant_item(self, ctx) -> int:
        item_id = await _add_item(ctx)
        async with ctx.session_maker() as db:
            for position, model in enumerate(("P1S", "X1C")):
                lib = LibraryFile(
                    filename=f"{model}.gcode.3mf",
                    file_path=f"/library/{model}.gcode.3mf",
                    file_size=10,
                    file_type="gcode.3mf",
                    file_metadata={"sliced_for_model": model},
                )
                db.add(lib)
                await db.flush()
                db.add(
                    PrintQueueVariant(
                        queue_item_id=item_id,
                        position=position,
                        library_file_id=lib.id,
                        target_model=model,
                        plate_id=position + 1,
                    )
                )
            await db.commit()
        return item_id

    @pytest.mark.asyncio
    async def test_picks_the_variant_for_the_printer_model(self, queue_db):
        item_id = await self._variant_item(queue_db)
        async with queue_db.session_maker() as db:
            item = await _load(db, item_id)
            assert variant_for_model(item, "X1C").target_model == "X1C"
            assert variant_for_model(item, "A1") is None
            assert variant_for_model(item, None) is None

    @pytest.mark.asyncio
    async def test_the_row_is_never_changed(self, queue_db):
        item_id = await self._variant_item(queue_db)
        before = await _get_item(queue_db, item_id)
        async with queue_db.session_maker() as db:
            item = await _load(db, item_id)
            copy = as_variant_item(item, variant_for_model(item, "X1C"))
            assert copy.plate_id == 2
            assert copy not in db
            assert not db.new
            assert item.library_file_id == before.library_file_id
            await db.commit()
        after = await _get_item(queue_db, item_id)
        assert (after.library_file_id, after.plate_id, after.target_model) == (
            before.library_file_id,
            before.plate_id,
            before.target_model,
        )

    @pytest.mark.asyncio
    async def test_no_variant_for_the_model_is_unknown(self, queue_db):
        item_id = await self._variant_item(queue_db)
        scheduler = PrintScheduler()
        with ExitStack() as stack:
            _patched(stack, _status(_tray(0, "PLA", "FFFFFFFF")))
            async with queue_db.session_maker() as db:
                item = await _load(db, item_id)
                fit = await fit_on_printer(db, item, 1, scheduler, printer_model="A1")
        assert fit.status == "unknown"
