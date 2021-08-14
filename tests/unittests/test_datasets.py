import csv
import shutil
import tempfile
import os


def test_to_csv():
    from speechbrain.dataio.datasets.lj import LJ

    def _get_fake_data():
        """
        Creates a LJ dataset from the included
        fake data for unit tests
        """
        module_path = os.path.dirname(__file__)
        data_path = os.path.join(module_path, "mockdata", "lj")
        return LJ(data_path)

    """
    Unit test for CSV creation
    """
    temp_dir = tempfile.mkdtemp()
    try:
        file_name = os.path.join(temp_dir, "test.csv")
        lj = _get_fake_data()
        lj.to_csv(file_name)
        with open(file_name) as csv_file:
            reader = csv.DictReader(csv_file)
            data = {row["ID"]: row for row in reader}
            item = data["LJ050-0159"]
            assert item["wav"].endswith("LJ050-0159.wav")
            assert item["label"].startswith("The Commission recommends")
            item = data["LJ050-0160"]
            assert item["wav"].endswith("LJ050-0160.wav")
            assert item["label"].startswith("The Commission further")
    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
