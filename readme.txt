=== The ARCADE Vessel Segmentation Challenge ===
== Submitted by ==
    Zvi Badash 214553034 zvi.badash@campus.technion.ac.il
    Adi Hatav 209143486 adi.hatav@campus.technion.ac.il

== Instructions == 
1) How to use?
    1.1) Create a virtual env using `python -m venv .venv`
    1.2) Activate it `source .venv/bin/activate`
    1.3) Install all dependencies using `pip install -r requirements.txt`
    1.4) Fetch the data and run a perliminary train cycle using `./fetch_data_and_train.sh`
        1.4.1) If does not work, try `chmod +x ./fetch_data_and_train.sh` first
        1.4.2) Return to 1.4
    1.5) You will be prompted to login to a W&B account
    1.6) The train session should begin now

2) How to reproduce our results?
    2.1) The hyper-parameters random search can bt ran using `python hpt-tuning.py`
    2.2) The final run of the model from which we creat our plots and evaluate performance is obtained using `python final_run.py`
    2.3) The main results and plots can be made by running the notebook `plots.ipynb` and `test_model.ipynb` * (See note)

3) Data
    3.1) All initial data handling can be found in `fetch_data.py` and the data URL is defined there



* Be sure to correctly set the `model_path` variable in the `test_model.ipynb` notebook which depends on the W&B run!