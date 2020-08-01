import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


class NearField:
    """ a class represent near field data """
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df_x, self.df_y = self.read_file()
        self.df = self.cal_intensity()

    def read_file(self):
        """
        read data from excel file, and create DataFrame
        :return:
        df1: intensity at x axis in near-field
        df2: intensity at y axis in near-firld
        """
        path = self.path
        file = self.file
        df = pd.read_excel(path + file)  # read excel file

        df1 = df.iloc[:, [0, 1]].dropna()  # choose columns
        df2 = df.iloc[:, [3, 4]].dropna()  # choose columns
        df1.columns = ['x_axis', 'intensity']  # rename columns
        df2.columns = ['y_axis', 'intensity']  # rename columns
        df1.loc[pd.isnull(df1['x_axis']), 'x_axis'] = 0
        df2.loc[pd.isnull(df2['y_axis']), 'y_axis'] = 0
        return df1, df2

    def cal_intensity(self):
        """
        calculate intensity
        :return:
        df3: calculated intensity with its corresponding x and y axis
        """
        df1, df2 = self.df_x, self.df_y

        df3 = pd.DataFrame()
        idx = 0

        # use 2 for loop to travel all of the x and y position
        for i, ax1 in enumerate(df1['x_axis']):
            for j, ax2 in enumerate(df2['y_axis']):
                df3 = df3.append(pd.DataFrame({
                    'x_axis': ax1 * 0.001,  # change unit from um to mm
                    'y_axis': ax2 * 0.001,  # change unit from um to mm
                    'intensity': np.square(df1['intensity'].iloc[i] * df2['intensity'].iloc[j])  # calculate intensity
                }, index=[idx]), ignore_index=True)
                idx += 1

        return df3

    def plot_intensity(self):
        """
        visualize intensities at near field
        """
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
    """ a class represent far field data """
    def __init__(self, path, file):
        self.path = path
        self.file = file
        self.df_l, self.df_v = self.read_file()
        self.df = self.cal_angle_intensity()

    def read_file(self):
        """
        read data from excel file, and create DataFrame
        :return:
        df1: intensity at l angle in far-field
        df2: intensity at v angle in far-firld
        """
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

    def cal_angle_intensity(self):
        """
        calculate intensity
        :return:
        df4: calculated intensity with its corresponding l and v angle
        """
        df1, df2 = self.df_l, self.df_v

        # use 2 for loop to travel all of the l and v angle
        df4 = pd.DataFrame()  # initialization
        idx = 0  # initialization
        for i, ax1 in enumerate(df1['L_angle']):
            for j, ax2 in enumerate(df2['V_angle']):
                alpha = np.cos(np.radians(df2['V_angle'].iloc[j])) * np.cos(np.radians(90 - df1['L_angle'].iloc[i]))
                beta = np.cos(np.radians(90 - df2['V_angle'].iloc[j]))
                gamma = np.sqrt(1 - np.square(alpha) - np.square(beta))

                df4 = df4.append(pd.DataFrame({
                    'L_angle': ax1,
                    'V_angle': ax2,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'intensity': np.square(df1['intensity'].iloc[i] * df2['intensity'].iloc[j])
                }, index=[idx]), ignore_index=True)
                idx += 1
        df4['intensity_ratio'] = df4['intensity'] / df4['intensity'].sum()

        return df4

    def plot_intensity(self):
        """
        visualize intensities at far field
        """
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
    """ a class represent source file """
    def __init__(self, df_near, df_far, trace_num_expect):
        """
        :param df_near: DataFrame which represent near field and created by the class NearField
        :param df_far: DataFrame which represent near field and created by the class FarField
        :param trace_num_expect: set your expected trace number (the exact trace number might be less than this number)
        """
        self.df_near = df_near
        self.df_far = df_far
        self.trace_num_expect = trace_num_expect
        self.df_far, self.trace_num_cell, self.trace_num = self.cal_trace_num()
        print('cell numbers in NF: %s, FF: %s, trace numbers per NF cell: %s, total trace numbers: %s'
              % (len(self.df_near), len(self.df_far), self.trace_num_cell, self.trace_num))

    def cal_trace_num(self):
        """
        calculate the exact trace number
        :return:
        df2: far field DataFrame with its corresponding trace number in each angle
        total_lines: total trace number
        """
        df1, df2 = self.df_near, self.df_far

        # distribute the total trace number to each cell, and round it as integer
        trace_num_per_cell = int(self.trace_num_expect / len(nf.df))

        # distribute the trace number in each cell to each angle in far field, and round it as integer
        df2['trace_num'] = list(map(int, df2['intensity_ratio'].values * trace_num_per_cell))

        # calculate the exact total trace number
        total_lines = df2['trace_num'].sum() * len(df1)

        return df2, df2['trace_num'].sum(), total_lines

    def cal_source_file(self):
        """
        create source file, and save as .txt file
        """
        df1, df2 = self.df_near, self.df_far
        df3 = df2[df2['trace_num'] > 0]  # filter out the rows with trace number = 0; df3 represent new far field data
        df3 = df3.reset_index(drop=True)  # reset index
        z_axis = 0

        times1 = dt.datetime.now()
        output = np.array([])  # initialize source file
        idx = 0

        # use 2 for loop to travel all cells with positive trace number
        # the 1st loop travel all cells in near field
        # the 2nd loop travel cells in far field from trace_num = 1 to max(trace_num)
        # note that trace_num represent the trace number in each cell in far-field
        for i in range(len(df1)):
            for j, trace_num in enumerate(df3['trace_num'].unique()):
                df4 = df3.loc[df3['trace_num'] == trace_num]  # select rows which trace_num == trace_num
                df4 = df4[['alpha', 'beta', 'gamma']]  # select alpha, beta and gamma

                # create array that represent 1 row of near-field and repeat "trace_num * len(df4)" times
                to_add_nf = np.tile(np.array([df1['x_axis'].iloc[i], df1['y_axis'].iloc[i], z_axis]),
                                    (trace_num * len(df4), 1))

                # create array that represent len(df4) rows of far-field data and repeat "trace_num" times
                to_add_ff = np.tile(df4.to_numpy(), (trace_num, 1))

                # create array that represent the intensity of this near-field and repeat "trace_num * len(df4)" times
                to_add_intensity = np.tile(np.array(df1['intensity'].iloc[i]), (trace_num * len(df4), 1))

                # combine above arrays
                to_add = np.hstack((to_add_nf, to_add_ff, to_add_intensity))

                # append the combined array to the output
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
        """
        set intensity threshold to decrease the calculation time
        if you set threshold = 0.01, that means let the smallest 1% intensity values be 0
        :param th_nf: threshold for near-field
        :param th_ff: threshold for far-field
        """
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
    sf = SourceFile(nf.df, ff.df, 3.5e6)
    sf.set_threshold(0.01, 0.01)
    sf.cal_source_file()
