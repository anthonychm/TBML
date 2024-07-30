import numpy as np

db = np.random.rand(50, 22)
isnan = np.isnan(db)
anyisnan = isnan.any()

assert np.isnan(db).any() == False
print('finish')