"""
This file contains methods for finding directories of data for data-driven RANS modelling.
"""


class DataDirectoryFinder:
    def __init__(self, case):
        self.case = case
        self.dir = None

    def get_data_dir(self):
        # Get directory for data of the given case ✓
        upper_name = "C:/Users/Antho/Dropbox (The University of Manchester)"
        if any(name in self.case for name in ["FBFS", "IMPJ"]):
            self.dir = upper_name + "/PhD_Anthony_Man/ML Database/" + self.case[:4]
        elif any(name in self.case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
            self.dir = upper_name + "/PhD_Anthony_Man/AM_PhD_shared_documents/" \
                                    "McConkey data"
        elif any(name in self.case for name in ["SQCY", "TACY"]):
            self.dir = upper_name + "/PhD_Anthony_Man/AM_PhD_shared_documents/Geneva data"
        else:
            raise Exception("No matching parent path for this case")
        return self.dir

    def get_path_dict(self, turb_model="komegasst"):
        if any(name in self.case for name in ["FBFS", "IMPJ"]):
            return {"rans": self.dir + '/' + self.case + '/',
                    "zonal": self.dir + '/zonal criteria/',
                    "labels": self.dir + '/' + self.case + '/'}
        elif any(name in self.case for name in ["BUMP", "CBFS", "CNDV", "PHLL", "DUCT"]):
            return {"rans": self.dir + '/' + turb_model + '/' + turb_model + '_',
                    "zonal": self.dir + '/' + turb_model + '/zonal criteria/',
                    "labels": self.dir + '/labels/'}
        else:
            raise Exception("No matching path dict for this case")

