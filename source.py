import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


class NearField:
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df_x, self.df_y = self.read_file()
        self.df = self.cal_intensity()

    def read_file(self):
        path = self.path
        file = self.file
        df = pd.read_excel(path + file)

        df1 = df.iloc[:, [0, 1]].dropna()
        df2 = df.iloc[:, [3, 4]].dropna()
        df1.columns = ['x_axis', 'intensity']
        df2.columns = ['y_axis', 'intensity']
        df1.loc[pd.isnull(df1['x_axis']), 'x_axis'] = 0
        df2.loc[pd.isnull(df2['y_axis']), 'y_axis'] = 0

        return df1, df2

    def cal_intensity(self):
        df1, df2 = self.df_x, self.df_y

        df3 = pd.DataFrame()
        idx = 0
        for i, ax1 in enumerate(df1['x_axis']):
            for j, ax2 in enumerate(df2['y_axis']):
                df3 = df3.append(pd.DataFrame({
                    'x_axis': ax1,
                    'y_axis': ax2,
                    'intensity': np.square(df1['intensity'].iloc[i] * df2['intensity'].iloc[j])
                }, index=[idx]), ignore_index=True)
                idx += 1

        return df3

    def plot_intensity(self):
        df1, df2 = self.df_x, self.df_y
        df3 = self.df

        intensity = df3['intensity'].to_numpy().reshape(len(df1['x_axis']), -1)
        plt.imshow(intensity, aspect='auto',
                   extent=[min(df2['y_axis']), max(df2['y_axis']), max(df1['x_axis']), min(df1['x_axis'])])
        plt.xlabel('Y axis')
        plt.ylabel('X axis')
        plt.title('Near Field')
        plt.savefig(path + 'Near Field.png')
        plt.close('all')


class FarField:
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df_l, self.df_v = self.read_file()
        self.df = self.cal_intensity()

    def read_file(self):
        path = self.path
        file = self.file
        df = pd.read_excel(path + file)
        df1 = df.iloc[:, [0, 1]].dropna()
        df2 = df.iloc[:, [3, 4]].dropna()
        df1.columns = ['L_angle', 'intensity']
        df2.columns = ['V_angle', 'intensity']
        df1.loc[pd.isnull(df1['L_angle']), 'L_angle'] = 0
        df2.loc[pd.isnull(df2['V_angle']), 'V_angle'] = 0

        return df1, df2

    def cal_intensity(self):
        df1, df2 = self.df_l, self.df_v

        df4 = pd.DataFrame()
        idx = 0
        for i, ax1 in enumerate(df1['L_angle']):
            for j, ax2 in enumerate(df2['V_angle']):
                df4 = df4.append(pd.DataFrame({
                    'L_angle': ax1,
                    'V_angle': ax2,
                    'x': np.sin(df1['L_angle'].iloc[i] * np.pi / 180) * np.cos(df2['V_angle'].iloc[j] * np.pi / 180),
                    'y': np.sin(df1['L_angle'].iloc[i] * np.pi / 180) * np.sin(df2['V_angle'].iloc[j] * np.pi / 180),
                    'z': np.cos(df1['L_angle'].iloc[i] * np.pi / 180),
                    'intensity': np.square(df1['intensity'].iloc[i] * df2['intensity'].iloc[j])
                }, index=[idx]), ignore_index=True)
                idx += 1
        df4['intensity_ratio'] = df4['intensity'] / df4['intensity'].sum()

        return df4

    def plot_intensity(self):
        df1, df2 = self.df_l, self.df_v
        df4 = self.df

        intensity = df4['intensity'].to_numpy().reshape(len(df1['L_angle']), -1)
        plt.imshow(intensity, aspect='auto',
                   extent=[min(df2['V_angle']), max(df2['V_angle']), max(df1['L_angle']), min(df1['L_angle'])])
        plt.xlabel('Vertical axis')
        plt.ylabel('Lateral axis')
        plt.title('Far Field')
        plt.savefig(path + 'Far Field.png')
        plt.close('all')


class SourceFile:
    def __init__(self, df_near, df_far, trace_num_expect):
        self.df_near = df_near
        self.df_far = df_far
        self.trace_num_expect = trace_num_expect
        self.df_far, self.trace_num_cell, self.trace_num = self.cal_trace_num()
        print('cell numbers in NF: %s, FF: %s, trace numbers per NF cell: %s, total trace numbers: %s'
              % (len(self.df_near), len(self.df_far), self.trace_num_cell, self.trace_num))

    def cal_trace_num(self):
        trace_num_per_cell = int(self.trace_num_expect / len(nf.df))

        df1, df2 = self.df_near, self.df_far
        df2['trace_num'] = list(map(int, df2['intensity_ratio'].values * trace_num_per_cell))
        total_lines = df2['trace_num'].sum() * len(df1)

        return df2, df2['trace_num'].sum(), total_lines

    def cal_source_file(self):
        df1, df2 = self.df_near, self.df_far
        df3 = df2[df2['trace_num'] > 0]
        df3 = df3.reset_index(drop=True)

        times1 = dt.datetime.now()
        output = np.array([])
        idx = 0
        for i in range(len(df1)):
            for j, trace_num in enumerate(df3['trace_num'].unique()):
                df4 = df3.loc[df3['trace_num'] == trace_num]
                df4 = df4[['x', 'y', 'z']]
                to_add_nf = np.tile(np.array([df1['x_axis'].iloc[i], df1['y_axis'].iloc[i]]), (trace_num * len(df4), 1))
                to_add_ff = np.tile(df4.to_numpy(), (trace_num, 1))
                to_add_intensity = np.tile(np.array(df1['intensity'].iloc[i]), (trace_num * len(df4), 1))
                to_add = np.hstack((to_add_nf, to_add_ff, to_add_intensity))
                output = np.vstack([output, to_add]) if output.size else to_add
                print('%s / %s' % (idx, self.trace_num))
                idx = idx + len(to_add)

        np.savetxt(path + 'source file.txt', output, delimiter='	', fmt='%.6f', header='%s 4' % self.trace_num)
        times2 = dt.datetime.now()
        T = times2 - times1
        T = T.total_seconds() / 60
        print('Spent %.0fmin' % T)
        print('Finished at ' + times2.strftime('%m/%d %H:%M'))

    def set_threshold(self, th_nf, th_ff):
        df1, df2 = self.df_near, self.df_far
        df1 = df1[df1['intensity'] > df1['intensity'].max() * th_nf].reset_index(drop=True)
        df2 = df2[df2['intensity'] > df2['intensity'].max() * th_ff].reset_index(drop=True)

        total_lines = df2['trace_num'].sum() * len(df1)
        self.df_near = df1
        self.df_far = df2
        self.trace_num_cell = df2['trace_num'].sum()
        self.trace_num = total_lines
        print('cell numbers in NF: %s, FF: %s, trace numbers per NF cell: %s, total trace numbers: %s'
              % (len(self.df_near), len(self.df_far), self.trace_num_cell, self.trace_num))


if __name__ == '__main__':
    path = 'C:/Users/cha75794/Desktop/Zemax/'
    file_near = 'Near-field data - temp.xlsx'
    file_far = 'Far-field data - temp.xlsx'

    nf = NearField(path, file_near)
    ff = FarField(path, file_far)
    sf = SourceFile(nf.df, ff.df, 100e6)
    sf.set_threshold(0.01, 0.01)
    sf.cal_source_file()
   
