from src.association import associate, head_region
from src.detector import HELMET, NO_HELMET, Detection

LEFT_PERSON = (0, 0, 100, 300)
RIGHT_PERSON = (200, 0, 300, 300)


def helmet(box, confidence=0.9, label=HELMET):
    return Detection(box, label, confidence, label)


def test_head_region_is_top_part_of_person():
    assert head_region((0, 0, 100, 200), 0.25) == (0, 0, 100, 50)


def test_helmet_goes_to_the_person_whose_head_it_is_on():
    evidence = associate([LEFT_PERSON, RIGHT_PERSON], [helmet((220, 5, 280, 60))])
    assert evidence[0].label is None
    assert evidence[1].label == HELMET
    assert evidence[1].confidence == 0.9


def test_helmet_not_on_a_head_is_ignored():
    # Helmet at waist height (e.g. carried in a hand).
    evidence = associate([LEFT_PERSON], [helmet((20, 200, 80, 250))])
    assert evidence[0].label is None
    assert not evidence[0].ambiguous


def test_helmet_between_two_overlapping_people_is_ambiguous():
    people = [(0, 0, 100, 300), (40, 0, 140, 300)]
    evidence = associate(people, [helmet((45, 5, 95, 50))])
    assert evidence[0].ambiguous and evidence[1].ambiguous
    assert evidence[0].label is None and evidence[1].label is None


def test_each_person_gets_their_own_status():
    evidence = associate(
        [LEFT_PERSON, RIGHT_PERSON],
        [helmet((20, 5, 80, 60)), helmet((220, 5, 280, 60), 0.8, NO_HELMET)],
    )
    assert evidence[0].label == HELMET
    assert evidence[1].label == NO_HELMET


def test_conflicting_evidence_of_similar_strength_is_ambiguous():
    evidence = associate(
        [LEFT_PERSON],
        [helmet((20, 5, 80, 60), 0.70), helmet((22, 6, 78, 58), 0.65, NO_HELMET)],
    )
    assert evidence[0].ambiguous


def test_clearly_stronger_evidence_wins():
    evidence = associate(
        [LEFT_PERSON],
        [helmet((20, 5, 80, 60), 0.95), helmet((22, 6, 78, 58), 0.50, NO_HELMET)],
    )
    assert evidence[0].label == HELMET
    assert not evidence[0].ambiguous
