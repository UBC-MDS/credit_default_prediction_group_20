# author: Althrun Sun
# date: 2022-11-23

"""
Doing EDA of the preprocessed data by drawing figures and table for visualization.

Usage:
eda_script.py --processed_data_path=processed_data_path --eda_result_path=eda_result_path

Options:
    --processed_data_path=<processed_data_path>       Local File Path where the processed data saved in.
    --eda_result_path=<eda_result_path>               Local File Path where the result of eda  saved in.
Example:
python ./src/eda_script.py --processed_data_path "./data/processed/credit_train_df.csv" --eda_result_path "./results"
"""

import os
from docopt import docopt
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
import warnings
import os
from altair_data_server import data_server
import matplotlib.pyplot as plt
import vl_convert as vlc
warnings.filterwarnings('ignore')
# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable('mimetype')
# Handle large data sets without embedding them in the notebook

alt.data_transformers.disable_max_rows()

opt = docopt(__doc__)


def main(processed_data_path, eda_result_path):
    """
    Driver function to get the output of eda
    and save it in the local file system.
    Parameters
    ----------
    processed_data_path : string
        Local File Path where the processed data saved in
    eda_result_path : string
        Local File Path where the result of eda  saved in.
    """
    def save_chart(chart, filename, scale_factor=1):
        '''
        Save an Altair chart using vl-convert
    
        Parameters
        ----------
        chart : altair.Chart
        Altair chart to save
        ilename : str
        The path to save the chart to
        scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
        '''
        if filename.split('.')[-1] == 'svg':
            with open(filename, "w") as f:
                f.write(vlc.vegalite_to_svg(chart.to_dict()))
        elif filename.split('.')[-1] == 'png':
            with open(filename, "wb") as f:
                f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
        else:
            raise ValueError("Only svg and png formats are supported")
    # Test if the URL exists and returns Status OK
    try:
        df=pd.read_csv(processed_data_path)
        print("successfully read the data")
    except :
        print("Unable to read data from the path.")
        return
    ##test assert
    assert df is not None
    ##create folder
    if not os.path.exists(eda_result_path+'/eda'):
        os.makedirs(eda_result_path+'/eda')
        os.makedirs(eda_result_path+'/eda/images')
        os.makedirs(eda_result_path+'/eda/eda_tables')
    ##test assert if folder created
    assert os.path.exists(eda_result_path+'/eda') is True
    # Read data
    credit_df=pd.read_csv(processed_data_path)
    #stat description
    describe_df=credit_df.describe()
    describe_df.to_csv('./'+eda_result_path+'/eda/eda_tables/describe_df.csv')
    ##test assert if table created
    assert describe_df is not None
    
    credit_df = credit_df.rename(columns={'default payment next month': 'default_payment_next_month'})
    # change target data type
    credit_df["default_payment_next_month"] = credit_df["default_payment_next_month"].astype("category")
    train_df, test_df = train_test_split(credit_df, test_size=0.2, random_state=522)
    #pie chart
    plt.pie(credit_df['default_payment_next_month'].value_counts(), 
        labels=['undefualt','default'],
        colors=["#d5695d", "#5d8ca8"],
        autopct='%.2f%%',)
    plt.title('proportion of target classes')
    plt.plot()
    plt.savefig('./'+eda_result_path+'/eda/images/target_proportion.jpg')
    
    ##test assert if pie chart
    assert plt is not None
    
    #heatmap
    corr_df = train_df.corr().stack().reset_index(name='corr')
    corr_plot=alt.Chart(corr_df,title="Correlation graph").mark_rect().encode(x='level_0',y='level_1',color=alt.Color('corr')).properties(width=400, height=400)
    save_chart(corr_plot,'./'+eda_result_path+'/eda/images/corr_plot.png', 2)
    
    ##test assert if heatmap created
    assert corr_plot is not None
    
    ##default-undefualt
    train_df['whether default next month'] = 'not defaulting'
    train_df.loc[train_df['default_payment_next_month'] == 1, 'whether default next month'] = 'defaulting'
    
    
    
    #numeric dis
    num_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", 
            "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3", 
            "PAY_AMT4","PAY_AMT5", "PAY_AMT6"]
    numeric_dis=alt.Chart(train_df).mark_bar().encode(
     alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=30)),
     y='count()',
    color=alt.Color('whether default next month', scale=alt.Scale( scheme='purpleorange'))).properties( width=200,height=150).repeat(num_cols,columns=3)
    save_chart(numeric_dis, './'+eda_result_path+'/eda/images/numeric_dis.png', 2)
    
    ##test assert if numeric_dis created
    assert numeric_dis is not None
 
    #categorical dis
    cat_col = ["EDUCATION", "MARRIAGE", "SEX", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    categorical_dis=alt.Chart(train_df).mark_bar().encode(
        alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=20)),y='count()',
      color=alt.Color('whether default next month', scale=alt.Scale( scheme='dark2'))
    ).properties(width=200,height=150).repeat(cat_col,columns=3)
    save_chart(categorical_dis,'./'+eda_result_path+'/eda/images/categorical_dis.png', 2)
    
    ##test assert if categorical dis created
    assert categorical_dis is not None
    
    
    
    print('output saved every things done')



    



# Execute only when run as a script.
if __name__ == "__main__":
    main(opt["--processed_data_path"],opt["--eda_result_path"])
    # main("./data/processed/credit_train_df.csv", "./results")