from . import register_skill, SkillParam


def _detect_objects_impl(source: str = "camera"):
    return {"objects": [{"id": 1, "type": "vessel", "bbox": [100, 200, 300, 400]}], "source": source}


def _track_objects_impl(objects=None, source: str = "camera"):
    return {"tracks": [{"id": 1, "position": [150, 250], "velocity": [2.1, 0.5]}], "source": source}


def _estimate_state_impl(track=None):
    return {"state": {"position": [150, 250], "velocity": [2.1, 0.5], "heading": 12.3}}


def _predict_path_impl(state=None, horizon: float = 5.0):
    return {"trajectory": [[150, 250], [152, 251], [155, 252]], "horizon": horizon}


def _assess_collision_impl(trajectory=None):
    return {"risk": 0.15, "time_to_collision": None, "safe": True}


def _generate_command_impl(state=None, trajectory=None):
    return {"command": {"thrust": 0.7, "steering": -0.2, "mode": "track"}}


def _localize_impl(landmarks=None):
    return {"pose": {"x": 10.5, "y": 20.3, "theta": 0.45}, "confidence": 0.94}


register_skill(
    name="detect_objects",
    func="detect_objects",
    category="perception",
    description="Detect objects in a sensor feed",
    params=[
        SkillParam(name="source", type_hint="string", description="Sensor source identifier"),
    ],
    implementation=_detect_objects_impl,
    llm_prompt_template="Detect and classify all objects visible in the {source} feed. Return bounding boxes and classifications.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["objects"],
        "properties": {
            "objects": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "type"],
                    "properties": {
                        "id": {"type": "integer"},
                        "type": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        },
    },
)

register_skill(
    name="track_objects",
    func="track_objects",
    category="perception",
    description="Track detected objects across frames",
    implementation=_track_objects_impl,
    llm_prompt_template="Track objects across consecutive frames. Maintain IDs and estimate velocities.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["tracks"],
        "properties": {
            "tracks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "position"],
                    "properties": {
                        "id": {"type": "integer"},
                        "position": {"type": "array", "items": {"type": "number"}},
                        "velocity": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        },
    },
)

register_skill(
    name="estimate_state",
    func="estimate_state",
    category="state_estimation",
    description="Estimate robot/target state from observations",
    implementation=_estimate_state_impl,
    llm_prompt_template="Estimate the kinematic state (position, velocity, heading) from tracking data.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["state"],
        "properties": {
            "state": {
                "type": "object",
                "required": ["position", "velocity"],
                "properties": {
                    "position": {"type": "array", "items": {"type": "number"}},
                    "velocity": {"type": "array", "items": {"type": "number"}},
                    "heading": {"type": "number"},
                },
            },
        },
    },
)

register_skill(
    name="predict_path",
    func="predict_path",
    category="planning",
    description="Predict future trajectory",
    implementation=_predict_path_impl,
    llm_prompt_template="Predict the object's trajectory over the next {horizon} seconds.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["trajectory"],
        "properties": {
            "trajectory": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "horizon": {"type": "number"},
        },
    },
)

register_skill(
    name="assess_collision",
    func="assess_collision",
    category="planning",
    description="Assess collision risk for a predicted trajectory",
    implementation=_assess_collision_impl,
    llm_prompt_template="Assess collision risk for the given trajectory. Return risk score and time-to-collision.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["risk", "safe"],
        "properties": {
            "risk": {"type": "number", "minimum": 0, "maximum": 1},
            "safe": {"type": "boolean"},
            "time_to_collision": {"type": "number"},
        },
    },
)

register_skill(
    name="generate_command",
    func="generate_command",
    category="control",
    description="Generate control command from state and plan",
    implementation=_generate_command_impl,
    llm_prompt_template="Generate a control command (thrust, steering) to achieve the planned trajectory.",
    uses_llm=True,
    output_schema={
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "object",
                "required": ["thrust", "steering"],
                "properties": {
                    "thrust": {"type": "number", "minimum": -1, "maximum": 1},
                    "steering": {"type": "number", "minimum": -1, "maximum": 1},
                    "mode": {"type": "string"},
                },
            },
        },
    },
)

register_skill(
    name="localize",
    func="localize",
    category="state_estimation",
    description="Localize robot position using landmarks",
    implementation=_localize_impl,
    uses_llm=False,
    output_schema={
        "type": "object",
        "required": ["pose", "confidence"],
        "properties": {
            "pose": {
                "type": "object",
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "theta": {"type": "number"},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
)
