# plots libraries
from matplotlib import pyplot as plt
import seaborn as sns

# manipulate data
import pandas as pd
import numpy as np
from seaborn.regression import statsmodels

# stats tools  ( include lineal model fit )
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy.stats as stats


class Regression_plots:

    def __init__(self, linear_model) -> None:
        self.linear_model = linear_model

    def residual_plot(self, ax=None):
        '''
        show the residual vs fitted values to diagnostic non linearity in data
        '''
        model_fitted_values = self.linear_model.fittedvalues
        model_fitted_values_resid = self.linear_model.resid
        model_fitted_values_resid.sort_values(ascending=False)
        resid_top_3 = abs(model_fitted_values_resid).sort_values(ascending=False)[:3]

        sns.residplot(ax=ax, x=model_fitted_values, y=model_fitted_values_resid, lowess=True,
                      scatter_kws={'alpha': 0.5}, line_kws={'color': 'k', 'lw': 1.5, 'alpha': 0.8})
        ax.set_xlabel("Fitted_values")
        ax.set_title("Residual vs Fitted values plot")
        ax.set_ylabel("Residual")

        for i in resid_top_3.index:
            ax.annotate(i, xy=(model_fitted_values[i], model_fitted_values_resid[i]))

    def q_q_plot(self, ax=None):
        sorted_student_residuals = pd.Series(
            self.linear_model.get_influence().resid_studentized_internal)
        sorted_student_residuals.index = self.linear_model.resid.index
        sorted_student_residuals = sorted_student_residuals.sort_values(ascending=True)

        df_student_residual = pd.DataFrame(sorted_student_residuals)
        df_student_residual.columns = ['sorted_student_residuals']
        df_student_residual['theoretical_quantiles'] = stats.probplot(
            df_student_residual['sorted_student_residuals'],
            dist='norm', fit=False)[0]

        rankings = abs(df_student_residual['sorted_student_residuals']).sort_values(ascending=False)
        student_residual_top_3 = rankings[:3]
        x = df_student_residual['theoretical_quantiles']
        y = df_student_residual['sorted_student_residuals']

        ax.scatter(x, y, edgecolor='k', facecolor='none')
        ax.set_title('Normal Q-Q')
        ax.set_ylabel('Standardized Residuals')
        ax.set_xlabel('Theoretical Quantiles')
        ax.plot([np.min([x, y]), np.max([x, y])], [
            np.min([x, y]), np.max([x, y])], color='r', ls='--')

        for val in student_residual_top_3.index:
            ax.annotate(val, xy=(df_student_residual['theoretical_quantiles'].loc[val],
                                 df_student_residual['sorted_student_residuals'].loc[val]))

    def scale_location_plot(self, ax=None):

        model_fitted_values = self.linear_model.fittedvalues
        student_residuals = self.linear_model.get_influence().resid_studentized_internal
        sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
        sqrt_student_residuals.index = self.linear_model.resid.index
        smoothed = lowess(sqrt_student_residuals, model_fitted_values)
        sqrt_student_residual_top3 = abs(sqrt_student_residuals).sort_values(ascending=False)[:3]
        ax.scatter(model_fitted_values, sqrt_student_residuals, edgecolors='k', facecolors='none')
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
        ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Scale-Location')
        ax.set_ylim(0, max(sqrt_student_residuals)+0.1)

        for i in sqrt_student_residual_top3.index:
            ax.annotate(i, xy=(model_fitted_values[i], sqrt_student_residuals[i]))

    def leverage_plot(self, ax=None):
        student_residuals = pd.Series(self.linear_model.get_influence().resid_studentized_internal)
        student_residuals.index = self.linear_model.resid.index
        df_leverage = pd.DataFrame(student_residuals)
        df_leverage.columns = ['student_residuals']
        df_leverage['leverage'] = self.linear_model.get_influence().hat_matrix_diag
        smoothed = lowess(df_leverage['student_residuals'], df_leverage['leverage'])
        sorted_student_residuals = abs(df_leverage['student_residuals']).sort_values(ascending=False)
        top3 = sorted_student_residuals[:3]
        leverage = abs(df_leverage['leverage']).sort_values(ascending=False)
        top_1_leverage = leverage[:1]

        x = df_leverage['leverage']
        y = df_leverage['student_residuals']
        xpos = max(x)+max(x)*0.01
        ax.scatter(x, y, edgecolors='k', facecolors='none')
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='r')
        ax.set_ylabel('Studentized Residuals')
        ax.set_xlabel('Leverage')
        ax.set_title('Residuals vs. Leverage')
        ax.set_ylim(np.min(y)-np.min(y)*0.15, max(y)+max(y)*0.15)
        ax.set_xlim(-0.01, max(x)+max(x)*0.08)
        ax.annotate(top_1_leverage.index[0], xy=(
            x.loc[top_1_leverage.index], y.loc[top_1_leverage.index]))
        for val in top3.index:
            ax.annotate(val, xy=(x.loc[val], y.loc[val]))

        cooksx = np.linspace(np.min(x), xpos, 50)
        p = len(self.linear_model.params)
        poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
        poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
        negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
        negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

        ax.plot(cooksx, poscooks1y, label="Cook's Distance", ls=':', color='r')
        ax.plot(cooksx, poscooks05y, ls=':', color='r')
        ax.plot(cooksx, negcooks1y, ls=':', color='r')
        ax.plot(cooksx, negcooks05y, ls=':', color='r')
        ax.plot([0, 0], ax.get_ylim(), ls=":", alpha=.3, color='k')
        ax.plot(ax.get_xlim(), [0, 0], ls=":", alpha=.3, color='k')
        ax.annotate('1.0', xy=(xpos, poscooks1y[-1]), color='r')
        ax.annotate('0.5', xy=(xpos, poscooks05y[-1]), color='r')
        ax.annotate('1.0', xy=(xpos, negcooks1y[-1]), color='r')
        ax.annotate('0.5', xy=(xpos, negcooks05y[-1]), color='r')
        ax.legend()

    def all_in_one(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        self.q_q_plot(ax=ax[0, 1])
        self.residual_plot(ax=ax[0, 0])
        self.scale_location_plot(ax=ax[1, 0])
        self.leverage_plot(ax=ax[1, 1])
        plt.show()


df = pd.read_csv('data/Auto.csv')

df.index = pd.RangeIndex(1, 398, 1)
df = df[df['horsepower'] != '?']
df['horsepower'] = pd.to_numeric(df['horsepower']).astype(np.int64)
linear_model = sm.OLS(df['mpg'], sm.add_constant(df['horsepower'])).fit()

reg = Regression_plots(linear_model)

reg.all_in_one()
