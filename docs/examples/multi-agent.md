# Multi-Agent Pipeline Example

Coordinate perception, planning, and control agents.

```ais
TASK autonomous_system {
    PERCEIVE {
        camera = camera_feed()
        lidar = lidar_scan()
    }

    REASON {
        objects = detect_objects(camera)
        path = plan_path(objects, lidar)
    }

    CONTROL {
        command = generate_command(path)
    }

    VERIFY {
        CHECK path.is_safe
        CHECK command.is_valid
    }
}
```
