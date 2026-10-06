from src.config import Config
from src.detector import HELMET, NO_HELMET, PERSON, map_class, model_labels, parse_detections


def test_class_names_are_mapped_case_insensitively():
    config = Config()
    assert map_class("person", config) == PERSON
    assert map_class("Hardhat", config) == HELMET
    assert map_class("NO-Hardhat", config) == NO_HELMET
    assert map_class("car", config) is None


def test_low_confidence_and_unwanted_classes_are_filtered():
    config = Config(person_confidence=0.5, helmet_confidence=0.45, no_helmet_confidence=0.5)
    names = {0: "person", 1: "Hardhat", 2: "NO-Hardhat", 3: "car"}
    boxes = [(0, 0, 10, 20), (1, 1, 5, 5), (2, 2, 6, 6), (0, 0, 9, 9), (3, 3, 8, 8)]
    confidences = [0.9, 0.46, 0.49, 0.99, 0.4]
    class_ids = [0, 1, 2, 3, 0]

    detections = parse_detections(boxes, confidences, class_ids, names, config,
                                  {PERSON, HELMET, NO_HELMET})

    assert [(det.label, round(det.confidence, 2)) for det in detections] == [
        (PERSON, 0.9),     # kept
        (HELMET, 0.46),    # above 0.45
    ]                      # no_helmet 0.49 < 0.5, car ignored, person 0.4 < 0.5


def test_only_allowed_labels_are_returned():
    config = Config()
    detections = parse_detections([(0, 0, 1, 1)], [0.99], [0], {0: "person"}, config, {HELMET})
    assert detections == []


def test_model_labels():
    config = Config()
    assert model_labels({0: "Hardhat", 1: "NO-Hardhat"}, config) == {HELMET, NO_HELMET}
