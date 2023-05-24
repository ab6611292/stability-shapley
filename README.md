Here you can find the code to reproduce experiments in the submission.
The results presented in the paper can be found in folders `census` and `titanic`.
Data from the tables are located in `<dataset>/res_<attack number>/f_vals.txt` and `<dataset>/res_<attack number>/g_vals.txt`, where `<dataset>` is either `census` or `titanic`.
The plots can be found in `census/res_<attack number>/images` and `titanic/res_<attack number>/images`.

To be able to run the experiments, you need to perform the following steps:
1. Download `adult.zip` file from https://www.kaggle.com/datasets/uciml/adult-census-income. Unzip the archive and place extracted `adult.csv` in folder `datasets/census`.
2. Download `train.csv` file from https://www.kaggle.com/competitions/titanic. Place the downloaded file into `datasets/titanic`
3. Create a python 3 environment and install all the requirements there with ```pip install -r requirements.txt```.
4. Run train_census.py: ```python train_census.py```.
5. Run train_titanic.py: ```python train_titanic.py```.
5. Run perform_atacks.py: ```python perform_attacks.py```.
6. Run plot_shapley.py: ```python plot_shapley.py```.
This script will create plots with Shapley values of the original and the manipulated models with both mean and median set functions.
The plots can be found in `census/res_<attack number>/images` and `titanic/res_<attack number>/images`.
7. Run plot_lipschitz_bound.py: ```python plot_lipschitz_bound.py```.
This script will plot Lipschitz constants. The plot can be found in project root.
