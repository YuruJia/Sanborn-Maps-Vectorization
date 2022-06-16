import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

array1 = [[15828452,121114,31802],
        [10462,1222720,197078],
        [3801,8323,270968]]
array2 = [[1222720,197078],
        [8323,270968]]
df_cm = pd.DataFrame(array2/np.sum(np.array(array2)), index = ["brick", "frame"],
                  columns = ["brick", "frame"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm,fmt='.2%', annot=True,annot_kws={"size": 16},square=True, cmap="YlGn")
plt.show()