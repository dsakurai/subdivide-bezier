from core import bezier_fit
import time
import pandas as pd

start_time = time.perf_counter()
names = ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP", "quantitative response, LC50 [-LOG(mol/L)]"]
df = pd.read_csv("resources/example-data/QSAR-fish/qsar_fish_toxicity.csv",
                 sep=";",
                 names=names
                 )
x = df[names[:-1]]
y = df[[names[-1]]]
x = x.values.tolist()
y = y.values.flatten().tolist()
testerr, caltime = bezier_fit(triangle=[], loop=5, max_degree=15, step=1,
                              datax=x, datay=y  # Load fish. (Comment this line to do this fitting with the default toy data)
                              )
avedf = pd.DataFrame(testerr)
timedf = pd.DataFrame(caltime)
avedf.to_csv("ave_err.csv",header=False, index=False, sep="\t")
timedf.to_csv("calc_time.csv",header=False, index=False, sep="\t")

for i in range(5):
    if i == 0:
        testerr, caltime = bezier_fit(triangle=[1], loop=5, max_degree=15, step=1,
                                      datax=x, datay=y  # Load fish. (Comment this line to do this fitting with the default toy data)
                                      )
        avedf = pd.DataFrame(testerr)
        timedf = pd.DataFrame(caltime)
        avedf.to_csv("ave_err.csv",header=False, index=False, sep="\t")
        timedf.to_csv("calc_time.csv",header=False, index=False, sep="\t")
    elif i == 1:
        testerr, caltime = bezier_fit(triangle=[1, 0], loop=5, max_degree=15, step=1, datax=x, datay=y)
        avedf = pd.DataFrame(testerr)
        timedf = pd.DataFrame(caltime)
        avedf.to_csv("ave_err0.csv",header=False, index=False, sep="\t")
        timedf.to_csv("calc_time0.csv",header=False, index=False, sep="\t")
    elif i == 2:
        testerr, caltime = bezier_fit(triangle=[1, 1], loop=5, max_degree=15, step=1, datax=x, datay=y)
        avedf = pd.DataFrame(testerr)
        timedf = pd.DataFrame(caltime)
        avedf.to_csv("ave_err1.csv",header=False, index=False, sep="\t")
        timedf.to_csv("calc_time1.csv",header=False, index=False, sep="\t")
    elif i == 3:
        testerr, caltime = bezier_fit(triangle=[1, 2], loop=5, max_degree=15, step=1, datax=x, datay=y)
        avedf = pd.DataFrame(testerr)
        timedf = pd.DataFrame(caltime)
        avedf.to_csv("ave_err2.csv",header=False, index=False, sep="\t")
        timedf.to_csv("calc_time2.csv",header=False, index=False, sep="\t")
    elif i == 4:
        testerr, caltime = bezier_fit(triangle=[1, 3], loop=5, max_degree=15, step=1, datax=x, datay=y)
        avedf = pd.DataFrame(testerr)
        timedf = pd.DataFrame(caltime)
        avedf.to_csv("ave_err3.csv",header=False, index=False, sep="\t")
        timedf.to_csv("calc_time3.csv",header=False, index=False, sep="\t")

#testriangle = []
#bezeir_fit(triangle=testriangle, datax=x, datay=y)
end_time = time.perf_counter()
print(end_time - start_time)

