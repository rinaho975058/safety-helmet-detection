from src.alert_manager import AlertManager
from src.helmet_classifier import Status


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(stable=3, cooldown=10.0, enabled=True):
    clock = FakeClock()
    sounds = []
    manager = AlertManager(stable, cooldown, enabled, sound=True, clock=clock,
                           sound_player=lambda: sounds.append(1))
    return manager, clock, sounds


def test_no_alert_before_violation_is_stable():
    manager, _, _ = make(stable=3)
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is None
    assert manager.update(1, Status.NO_HELMET, 2, 0.9) is None
    assert manager.update(1, Status.NO_HELMET, 3, 0.9) is not None


def test_one_continuous_violation_alerts_once():
    manager, clock, sounds = make(stable=1)
    alerts = []
    for frame in range(1, 200):
        clock.now = frame * 0.05
        alerts.append(manager.update(1, Status.NO_HELMET, frame, 0.9))
    assert sum(alert is not None for alert in alerts) == 1
    assert len(sounds) == 1


def test_helmet_and_unknown_never_alert():
    manager, _, _ = make(stable=1)
    assert manager.update(1, Status.HELMET, 50, 0.9) is None
    assert manager.update(1, Status.UNKNOWN, 50, 0.9) is None


def test_cooldown_blocks_a_quick_repeat_then_allows_a_new_event():
    manager, clock, _ = make(stable=1, cooldown=10)
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is not None

    clock.now = 2
    manager.update(1, Status.HELMET, 1, 0.9)                     # put helmet on
    clock.now = 4
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is None   # took it off again: within cooldown

    clock.now = 20
    manager.update(1, Status.HELMET, 1, 0.9)
    clock.now = 21
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is not None


def test_different_people_alert_separately():
    manager, _, _ = make(stable=1)
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is not None
    assert manager.update(2, Status.NO_HELMET, 1, 0.9) is not None
    assert manager.total_alerts == 2


def test_disabled_alerts_do_not_fire_later_for_same_event():
    manager, _, sounds = make(stable=1)
    manager.enabled = False
    assert manager.update(1, Status.NO_HELMET, 1, 0.9) is None
    manager.enabled = True
    assert manager.update(1, Status.NO_HELMET, 2, 0.9) is None   # same continuous event
    assert sounds == []
