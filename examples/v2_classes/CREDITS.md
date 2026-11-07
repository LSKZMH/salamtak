# Photo credits (v2-class demo examples)

Five held-out photos that show the classes added in v2: vest, machinery, vehicle and utility pole.
They come from the **validation split** of
[Construction Hazard Detection](https://universe.roboflow.com/object-detection-qn97p/construction-hazard-detection)
(Roboflow workspace `object-detection-qn97p`, version 97), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0). Validation images were never used for training.
Workers are small or seen from behind; no faces are identifiable.

| file | dataset file (our v2 val alias) | licence | changes | deployed v2n finds (conf ≥ 0.35) |
|---|---|---|---|---|
| `concrete_pour_street.jpg` | chd_train_000475 | CC BY 4.0 | resized to 1600 px wide | machinery, 2 vehicles, 3 people, vest, helmet |
| `excavator_fence_workers.jpg` | chd_train_000292 | CC BY 4.0 | resized to 1600 px wide | machinery, vehicle, 2 people, 2 vests |
| `mixer_truck_utility_pole.jpg` | chd_train_000385 | CC BY 4.0 | resized to 1600 px wide | machinery, vehicle, utility pole |
| `bucket_truck_power_pole.jpg` | chd_train_006579 | CC BY 4.0 | resized to 1600 px wide | machinery, utility pole, 2 people, 2 vests, helmet |
| `earthworks_dump_trucks.jpg` | chd_train_002129 | CC BY 4.0 | resized to 1600 px wide | 3 machinery, utility pole |
