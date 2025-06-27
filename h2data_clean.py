# Save this as h2data_clean.py
import pandas as pd
import pymatgen
from pymatgen.core.composition import Composition

class H2DataClean:
    def __init__(self, fname, remove_nan_attr, remove_classes, headerlength=6):
        self.fname = fname
        self.remove_nan_attr = remove_nan_attr
        self.remove_classes = remove_classes
        self.headerlength = headerlength
        self._database = pd.read_csv(self.fname, header=self.headerlength)

    def clean_nans(self):
        for at in self.remove_nan_attr:
            self._database = self._database[pd.to_numeric(self._database[at], errors='coerce').notnull()]

    def clean_classes(self):
        for c in self.remove_classes:
            self._database = self._database[self._database["Material_Class"] != c]

    def clean_Mm(self):
        self._database = self._database[~self._database["Composition_Formula"].str.contains("Mm")]
        self._database = self._database[~self._database["Composition_Formula"].str.contains("Lm")]

    def clean_wt_percent_compositions(self):
        self._database = self._database[~self._database["Composition_Formula"].str.contains("-")]

    def clean_composition_formula(self):
        # Optionally: store original
        self._database["Original_Composition_Formula"] = self._database["Composition_Formula"]
        self._database["Composition_Formula"] = self._database["Composition_Formula"].apply(self.pymatgen_reduce_composition)
        self._database = self._database[~self._database["Composition_Formula"].str.contains("-")]
        self._database = self._database[~self._database["Composition_Formula"].str.contains("\+")]
        self._database = self._database[~self._database["Composition_Formula"].str.contains("M1.")]
        # Optionally remove H from complex hydrides if present

    def pymatgen_reduce_composition(self, formula):
        try:
            formula = formula.split()[0]
            c = Composition(formula)
            return c.reduced_formula
        except:
            return formula  # fallback, keep as is

    def stepwise_clean(self):
        print("Initial rows:", len(self._database))
        self.clean_nans()
        print("After NaN cleaning:", len(self._database))
        self.clean_classes()
        print("After class removal:", len(self._database))
        self.clean_Mm()
        print("After Mischmetal removal:", len(self._database))
        self.clean_wt_percent_compositions()
        print("After wt% formula removal:", len(self._database))
        self.clean_composition_formula()
        print("After composition formula cleaning:", len(self._database))

    def get_cleaned(self):
        return self._database
