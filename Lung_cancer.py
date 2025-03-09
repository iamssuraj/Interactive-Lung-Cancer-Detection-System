import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st
from tensorflow import keras
from imblearn.combine import SMOTETomek
from streamlit_marquee import streamlit_marquee

st.set_page_config(page_title="Lung Cancer Detection", page_icon=":hospital:", layout="wide")
st.markdown("""
<style>
.title {
    text-align: center;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 25px;
}
.centerer {
    text-align: center;
}
.marquee-container {
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    background: #FFFFFF;
    color: #000000;
    font-size: 16px;
    padding: 10px;
}
.marquee-content {
    display: inline-block;
    animation: marquee 15s linear infinite;
}
@keyframes marquee {
    from { transform: translateX(100%); }
    to { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)
#streamlit title
st.markdown('<h1 class="title">Lung Cancer Detection</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="marquee-container">
        <div class="marquee-content">
            Machine learning models may occasionally produce false positives, hence, it is recommended to not rely on the results predicted here.
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

convert = {
    'Yes': 2,
    'No': 1,
    'M': 1,
    'F': 0,
    1: 'High chances of Lung Cancer',
    0: 'Low chances of Lung Cancer'
}
# Yes - 2, No - 1


def main():
    
    #data retrieval
    data = pd.read_csv('cancer_data.csv')

    #plotting label
    # label_counts = data['LUNG_CANCER'].value_counts()
    # fig, ax = plt.subplots()
    # colors = ['Red', 'Green']
    # ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=colors)
    # st.pyplot(fig)

    data['LUNG_CANCER'] = data['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})
    data['GENDER'] = data['GENDER'].replace({'M': 1, 'F': 0})

    X = data.drop(['LUNG_CANCER'], axis = 1)
    Y = data['LUNG_CANCER'].astype(int)


    #Handling imbalance
    smk = SMOTETomek(random_state = 42)
    X, Y = smk.fit_resample(X, Y)
    data = X
    data['LUNG_CANCER'] = Y
    X = data.drop(['LUNG_CANCER'], axis = 1)
    Y = data['LUNG_CANCER'].astype(int)
    

    if 'state_check' not in st.session_state:
        st.session_state.state_check = False

    if st.button("View Dataset"):
        # Toggle the state_check state when the button is clicked
        st.session_state.state_check = not st.session_state.state_check
        st.rerun() 
        
    if st.session_state.state_check:
        st.dataframe(data)  # Display data as a dataframe


    #Parameters
    st.write("### Parameters: ")
    k = list(data.columns)
    st.markdown(", ".join(k))
    user_inputs = []
    z = False
    fill_random = st.radio("Choose data entry type", ("None", "Manual entry", "Examples"), horizontal = True)
    if fill_random == "Examples":
        num = st.number_input("There are 10 example inputs, type a number between 1 to 10", step = 1)
        
        example_inputs_set = [
            [0, 11, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2],
            [1, 79, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [0, 18, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1],
            [0, 36, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
            [1, 13, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1],
            [1, 43, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1],
            [1, 54, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1],
            [0, 60, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2],
            [1, 32, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1],
            [0, 66, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1]
        ]
        if(num >= 1 and num <= 10):
            user_inputs = example_inputs_set[num-1]
            st.text(user_inputs)
            z = True
        elif num:
            st.error("Please type number between 1 and 10 only")
    elif fill_random == "Manual entry":
        gender = st.selectbox("Please select your gender : ", ("Select", "M", "F"))
        user_inputs.append(convert.get(gender))
        age = st.number_input("Please enter your age : ", value=0, step=1)
        user_inputs.append(age)
        smoke = st.selectbox("Do you smoke?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(smoke))
        yellow_fingers = st.selectbox(
            "Do you have yellow fingers?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(yellow_fingers))
        anxiety = st.selectbox("Do you have anxiety issues?",
                            ("Select", "Yes", "No"))
        user_inputs.append(convert.get(anxiety))
        chronic = st.selectbox(
            "Do you have any chronic diseases?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(chronic))
        fatigue = st.selectbox("Do you experience fatigue?",
                            ("Select", "Yes", "No"))
        user_inputs.append(convert.get(fatigue))
        allergy = st.selectbox("Do you have any allergies?",
                            ("Select", "Yes", "No"))
        user_inputs.append(convert.get(allergy))
        wheezing = st.selectbox("Do you wheeze?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(wheezing))
        alcohol = st.selectbox("Do you consume alcohol?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(alcohol))
        cough = st.selectbox("Do you cough on a regular basis?",
                            ("Select", "Yes", "No"))
        user_inputs.append(convert.get(cough))
        breathing = st.selectbox(
            "Do you have any kind of difficulity in breathing?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(breathing))
        swallowing = st.selectbox(
            "Do you have any kind of difficulity in swallowing?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(swallowing))
        chest_pain = st.selectbox(
            "Are you suffering from chest pain?", ("Select", "Yes", "No"))
        user_inputs.append(convert.get(chest_pain))
        if not (gender and age and smoke and yellow_fingers and anxiety and chronic and fatigue and allergy and wheezing and alcohol and cough and breathing and swallowing and chest_pain) or None in user_inputs:
            st.error("Please fill all the fields")
        else:
            z = True
    
    # model handling
    if z:
        selected_model = st.selectbox("Select your model : ", ("None", "Logistic Regression", "Decision Tree", "Support Vector Machine",
                                                            "K Nearest Neigbours", "Random Forest", "Naive Bayes", "Artificial Neural Networks"))
        
        if not selected_model or selected_model == "None":
            pass
        else:
            # st.write('Model : ', selected_model)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
            # conversion of array
            narry = np.asarray(user_inputs, dtype=int)
            reshaped_array = narry.reshape(1, -1)

            if selected_model == "Logistic Regression":
                solver = st.selectbox("select solver", ("None", "liblinear", "lbfgs", "sag", "saga"))
                if solver and solver != "None":
                    if solver == "liblinear":
                        penalty = st.selectbox("Please select penalty", ('l1', 'l2'))
                    elif solver == "lbfgs" or solver == "sag":
                        penalty = st.selectbox("Please select penalty", ('l2', 'none'))
                    else:
                        penalty = st.selectbox("Please select penalty", ('l1', 'l2', 'none'))
                    Logistic_Regression_model = LogisticRegression(max_iter=1000, solver=solver, penalty=penalty)
                    Logistic_Regression_model.fit(X_train.values, Y_train.values)
                    st.write("Prediction using Logistic regression : ", convert.get(
                        Logistic_Regression_model.predict(reshaped_array)[0]))
                    ptrain = Logistic_Regression_model.predict(X_train.values)

                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                    ptest = Logistic_Regression_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} {round(accuracy*100, 5)}%")

            elif selected_model == "Decision Tree":
                criterion = st.selectbox("Criterion", ('Select', 'gini', 'entropy'))
                if criterion != 'Select':
                    Decision_Tree_model = DecisionTreeClassifier(criterion=criterion)
                    Decision_Tree_model.fit(X_train.values, Y_train.values)
                    st.write("Prediction using Decision Tree : ", convert.get(
                        Decision_Tree_model.predict(reshaped_array)[0]))
                    view_tree = st.button("View Tree")
                    if view_tree:
                        # plt.figure(figsize=(width, height))
                        with st.spinner('Please wait while the decision tree is being loaded...'):
                            plt.figure(figsize=(50, 10))
                            plot_tree(decision_tree=Decision_Tree_model, filled=True,
                                feature_names=list(X.columns), rounded=True)
                            st.pyplot(plt)            
                    ptrain = Decision_Tree_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                    
                
                    ptest = Decision_Tree_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                    
            elif selected_model == "Support Vector Machine":
                # svm_model = SVC() #By default Radial - Basis function kernel
                selected_kernel = st.selectbox(
                    "Please select a kernel", ("None", "linear", "rbf", "poly"))
                if selected_kernel and selected_kernel != "None":
                    svm_model = SVC(kernel=selected_kernel)
                    svm_model.fit(X_train.values, Y_train.values)
                    st.write("Prediction using SVM Model : ", convert.get(
                        svm_model.predict(reshaped_array)[0]))
                    ptrain = svm_model.predict(X_train.values)
                    accuracy = accuracy_score(Y_train, ptrain)
                    st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                    
                
                    ptest = svm_model.predict(X_test.values)
                    accuracy = accuracy_score(Y_test, ptest)
                    st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")

            elif selected_model == "K Nearest Neigbours":
                # knn_model = KNeighborsClassifier(value of k) #By default, value of k is 5
                weights = st.selectbox("select weights", ("Select", "uniform", "distance"))
                if weights != "Select":
                    kval = st.slider("Please select the value of k", 0, len(X_train.values))
                    if kval:
                        knn_model = KNeighborsClassifier(n_neighbors = kval, weights=weights)
                        knn_model.fit(X_train.values, Y_train.values)
                        st.write("Prediction using KNN Model : ", convert.get(
                            knn_model.predict(reshaped_array)[0]), " with k as ", kval)
                        ptrain = knn_model.predict(X_train.values)
                        accuracy = accuracy_score(Y_train, ptrain)
                        st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                        
                    
                        ptest = knn_model.predict(X_test.values)
                        accuracy = accuracy_score(Y_test, ptest)
                        st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")

            elif selected_model == "Random Forest":
                n = st.slider("select number of estimators", 1, 5000)
                criterion = st.selectbox("Criterion", ('gini', 'entropy'))
                Random_Forest_Classifier = RandomForestClassifier(n_estimators = n, criterion=criterion)
                Random_Forest_Classifier.fit(X_train.values, Y_train.values)
                st.write("Prediction using Random Forest Classifier : ", convert.get(
                        Random_Forest_Classifier.predict(reshaped_array)[0]))
                
                ptrain = Random_Forest_Classifier.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                
                ptest = Random_Forest_Classifier.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                
            elif selected_model == "Naive Bayes":
                Naive_Bayes_model = GaussianNB()
                Naive_Bayes_model.fit(X_train.values, Y_train.values)
                st.write("Prediction using Naive Bayes Model : ", convert.get(
                    Naive_Bayes_model.predict(reshaped_array)[0]))
                
                ptrain = Naive_Bayes_model.predict(X_train.values)
                accuracy = accuracy_score(Y_train, ptrain)
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")
                
                ptest = Naive_Bayes_model.predict(X_test.values)
                accuracy = accuracy_score(Y_test, ptest)
                st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")
            elif selected_model == "Artificial Neural Networks":
                model = keras.Sequential([
                    keras.layers.Dense(100, input_shape=(14,), activation = 'sigmoid'),
                    # keras.layers.Dropout(0.2),
                    keras.layers.Dense(100, activation = 'sigmoid'),
                    # keras.layers.Dropout(0.3),
                    keras.layers.Dense(100, activation = 'sigmoid'),
                    # keras.layers.Dropout(0.2),
                    keras.layers.Dense(2, activation = 'sigmoid'),
                ])
                model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
                with st.spinner('Please wait while model is running'):
                    history = model.fit(X_train, Y_train, epochs = 120, verbose=0)
                predictArray = [reshaped_array]
                predictAns = model.predict(predictArray)
                st.write("Prediction using Artificial Neural networks : ", convert.get(np.argmax(predictAns)))
                
                accuracy = history.history['accuracy'][-1]
                st.write(f"Train accuracy for {selected_model} is {round(accuracy*100, 5)}%")

            
                loss, accuracy = model.evaluate(X_test, Y_test)
                st.write(f"Test accuracy for {selected_model} is {round(accuracy*100, 5)}%")

if __name__ == "__main__":
    main()
