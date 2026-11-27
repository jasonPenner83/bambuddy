"""GET /queue/fit: how each pending job fits its printers' loaded filament.

Tray readings and the 3MF parse are stood in for; the matcher, the printer
candidates and the access rules are the real ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.print_scheduler import scheduler
from backend.tests.integration import test_queue_todo_3615 as todo_tests
from backend.tests.integration.test_queue_todo_3615 import _auth, make_item  # noqa: F401 - fixture

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

URL = "/api/v1/queue/fit"
WHITE_PLA = [{"slot_id": 1, "type": "PLA", "color": "#FFFFFF", "tray_info_idx": "", "used_grams": 20.0}]


def _status(tray_type: str, color: str, state: str = "IDLE") -> SimpleNamespace:
    tray = {"id": "0", "tray_type": tray_type, "tray_color": color, "tray_info_idx": "", "remain": 80}
    return SimpleNamespace(
        raw_data={"ams": [{"id": "0", "tray": [tray]}], "vt_tray": [], "ams_extruder_map": {}},
        nozzles=[],
        fila_switch=None,
        ams_filament_backup=None,
        state=state,
    )


@pytest.fixture
def loaded():
    """Per-printer tray readings: ``loaded[printer_id] = _status(...)``."""
    readings: dict[int, SimpleNamespace] = {}

    def get_status(pid):
        return readings.get(pid)

    with (
        patch("backend.app.services.printer_manager.printer_manager.get_status", MagicMock(side_effect=get_status)),
        patch("backend.app.services.print_scheduler.printer_manager.get_status", MagicMock(side_effect=get_status)),
        patch("backend.app.services.printer_manager.printer_manager.is_connected", MagicMock(return_value=True)),
        patch.object(
            scheduler,
            "_get_filament_requirements",
            AsyncMock(side_effect=lambda db, item: [dict(r) for r in WHITE_PLA]),
        ),
        patch.object(scheduler, "_short_filaments_on", AsyncMock(return_value=[])),
    ):
        yield readings


class TestFit:
    async def test_empty_queue(self, async_client, printer_factory, loaded):
        await printer_factory()
        assert (await async_client.get(URL)).json() == {"jobs": []}

    async def test_pinned_job_lists_its_printer(self, async_client, printer_factory, make_item, loaded):
        a = await printer_factory(name="A", model="X1C")
        item = await make_item(a.id, name="Benchy")
        loaded[a.id] = _status("PLA", "FFFFFFFF")

        body = (await async_client.get(URL)).json()

        assert body["jobs"] == [
            {
                "id": item.id,
                "name": "Benchy",
                "printer_id": a.id,
                "target_model": None,
                "printers": [
                    {
                        "printer_id": a.id,
                        "printer_name": "A",
                        "status": "fits",
                        "online": True,
                        "idle": True,
                        "swaps_needed": 0,
                        "unmatched": [],
                        "short": [],
                        "color_substitutions": [],
                    }
                ],
            }
        ]

    async def test_model_job_lists_every_printer_of_its_model_best_first(
        self, async_client, printer_factory, make_item, loaded
    ):
        petg = await printer_factory(name="PETG one", model="X1C")
        pla = await printer_factory(name="PLA one", model="X1C")
        other_model = await printer_factory(name="P1S", model="P1S")
        await make_item(None, name="Any", target_model="X1C")
        loaded[petg.id] = _status("PETG", "FFFFFFFF")
        loaded[pla.id] = _status("PLA", "000000FF")
        loaded[other_model.id] = _status("PLA", "FFFFFFFF")

        printers = (await async_client.get(URL)).json()["jobs"][0]["printers"]

        assert [(p["printer_name"], p["status"]) for p in printers] == [("PLA one", "fits"), ("PETG one", "unmatched")]
        assert printers[0]["color_substitutions"][0]["got"] == "#000000"
        assert printers[1]["unmatched"][0]["type"] == "PLA"
        assert printers[1]["swaps_needed"] == 1

    async def test_printer_without_a_reading_is_unknown(self, async_client, printer_factory, make_item, loaded):
        a = await printer_factory(name="A")
        await make_item(a.id)
        printers = (await async_client.get(URL)).json()["jobs"][0]["printers"]
        assert printers[0]["status"] == "unknown"

    async def test_only_pending_jobs(self, async_client, printer_factory, make_item, loaded):
        a = await printer_factory()
        await make_item(a.id, status="printing")
        assert (await async_client.get(URL)).json()["jobs"] == []

    async def test_nothing_is_written(self, async_client, db_session, printer_factory, make_item, loaded):
        from backend.app.models.print_queue import PrintQueueItem

        a = await printer_factory()
        item = await make_item(a.id)
        loaded[a.id] = _status("PLA", "FFFFFFFF")
        await async_client.get(URL)
        await db_session.refresh(item)
        row = await db_session.get(PrintQueueItem, item.id)
        assert (row.ams_mapping, row.waiting_reason, row.color_substitutions, row.manual_start) == (
            None,
            None,
            None,
            False,
        )


class TestFitAccess:
    """Same visibility as the To do list: queue read permission and printer scope."""

    async def _setup(self, async_client, permissions, printer_ids=None):
        return await todo_tests.TestTodoAccess()._setup(async_client, permissions, printer_ids)

    async def test_403_without_queue_read(self, async_client, printer_factory):
        await printer_factory()
        _, member = await self._setup(async_client, ["printers:read"])
        assert (await async_client.get(URL, headers=_auth(member))).status_code == 403

    async def test_scope_hides_other_printers(self, async_client, printer_factory, make_item, loaded):
        a = await printer_factory(name="A", model="X1C")
        b = await printer_factory(name="B", model="X1C")
        await make_item(a.id, name="OnA")
        await make_item(b.id, name="OnB")
        await make_item(None, name="AnyX1C", target_model="X1C")
        _admin, member = await self._setup(async_client, ["queue:read_all"], printer_ids=[a.id])

        body = (await async_client.get(URL, headers=_auth(member))).json()

        names = {j["name"] for j in body["jobs"]}
        assert names == {"OnA", "AnyX1C"}
        assert {p["printer_id"] for j in body["jobs"] for p in j["printers"]} == {a.id}
