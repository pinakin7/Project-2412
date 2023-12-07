import csv

class CSVLogger:
    def __init__(self, filename, fid=False):
        self.filename = filename
        self.fid = fid
        if fid:
            self.fieldnames = ['data_origin', 'num_images', 'FID']
        else:
            self.fieldnames = ['data_origin', 'image_number', 'predicted_distribution']

        self._create_csv()

    def _create_csv(self):
        with open(self.filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, model, image_number, predicted_distribution):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if not self.fid:
                writer.writerow({
                    'data_origin': model,
                    'image_number': image_number,
                    'predicted_distribution': predicted_distribution
                })
            if self.fid:
                writer.writerow({
                    'data_origin': model,
                    'num_images': image_number,
                    'FID': predicted_distribution
                })

if __name__ == "__main__":
    # Example usage:
    logger = CSVLogger('A:\\Project\\2412\\logs\\log.csv')

    # Log some data
    logger.log('Model1', 1, [0.1, 0.4, 0.5])
    logger.log('Model2', 2, [0.2, 0.3, 0.5])
