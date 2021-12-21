# science tools
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


def create_fake_data():
    tmin = 0
    tmax = 5
    period = 1
    nstep = 1000
    times = np.linspace(tmin, tmax, nstep)

    def fake_model(times, period, beta):
        return np.sin(times*period*2*np.pi) * np.exp(-beta*times) + 1
    # x(t) = sin(wt) exp(-bt) + 1
    # "vectorization numpy" efficienza di calcolo sin di un array

    values = fake_model(times, period, 1)

    # data viz
    if 0:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(times, values)
        plt.show()
        plt.savefig("data.png")
        plt.close()

    # import dati esterni

    classes_file = "classes.json"
    with open(classes_file) as jin:
        classes = json.load(jin)

    # print(classes)
    # print(classes["small"])
    # print(classes["small"]["beta"])
    # print(classes.keys())
    # print(classes.values())
    # print(classes.items())

    # "python generator/comprehension lists"
    betas = [(k, v["beta"]) for k, v in classes.items()]
    vals = np.array(fake_model(times, period, v["beta"]) for k, v in classes.items())
    print(vals)

    fakedata_file = "data.csv"
    with open(fakedata_file, "w") as fout:
        fout.write("class, beta, time, value\n")
        for (cl, beta), vs in zip(betas, vals):
            for t, v in zip(times, vs):
                fout.write(f"{cl},{beta},{t},{v}\n")


def process_data():
    if 0:
        fakedata_file = "data.csv"
        df = pd.read_csv(fakedata_file, sep=",")
        # print(df.time) type introspection
        # stats = df.groupby("class").count()
        # stats = df.groupby("class").sum()
        # stats = df.groupby("class").mean()
        # stats = df.groupby("class").std()
        stats = df.groupby("class").agg({
            "beta": "first",
            "time": "max",
            "value": "mean",
        })
        print(stats)
    table = df.set_index(["class", "time"]).drop(columns="beta")
    table = table.unstack(level=0)
    table.columns = table.columns
    print(table)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    table.plot(ax=ax)
    plt.show()
    plt.savefig("process.png")
    plt.close()




if __name__ == "__main__":
    create_fake_data()
    process_data()
