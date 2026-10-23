def robocasa_dataset_transform(trajectory):
    trajectory["action"] = trajectory["action"][..., :7]
    trajectory["observation"]["state"] = trajectory["observation"]["state"][..., :25]
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {"robocasa_no_noops": robocasa_dataset_transform}
