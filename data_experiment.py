from ROI import ROI


class ExperimentSettings:

    ROI_values = []

    def add_ROI_values(roi):
        ExperimentSettings.ROI_values.append(roi)

    def delete_ROI_value(index):
        ExperimentSettings.ROI_values.pop(index)