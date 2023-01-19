import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


def get_dataframe(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return df


def adjust_dataframe(df):
    p = []
    for i in range(len(df)):
        x = df.loc[i, 'graph']
        P = x[3:-9]
        new_P = P.replace("_", "")
        p.append(new_P)

    df['Prob'] = p
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def read_txt_file(path):
    file = open(path, 'r')
    Lines = file.readlines()
    g = []
    v = []
    e = []
    c = []
    t = []
    for i in range(len(Lines)):
        if 'Working' in Lines[i]:
            s = Lines[i + 3]
            t.append(s[7:-1])
            s = Lines[i + 4]
            c.append(s[16:-1])
            s = Lines[i + 5]
            g.append(s[7:-5])
            s = Lines[i + 6]
            v.append(s[20:-1])
            s = Lines[i + 7]
            e.append(s[17:-1])
    print(g)

    df = pd.DataFrame()
    df['graph'] = g
    df['cuts'] = c
    df['vertices'] = v
    df['edges'] = e
    df['time'] = t

    return df


def dataframe_statistics(df, ref_column, relevant_columns):
    df_statistics = df[relevant_columns].groupby([ref_column]).describe()
    return df_statistics

