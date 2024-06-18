import pandas as pd
import numpy as np




cabin_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
                 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20,
                 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}

# Convert the Cabin name to a number
def convert_cabin(cabin):
    if cabin == 0:
        return 0
    
    letter = cabin[0]

    return cabin_mapping.get(letter, 0)

def clean_data(entryPath):
    # Load data csv
    df = pd.read_csv(entryPath, index_col='PassengerId')
    
    # One hot encode Sex
    modData = pd.get_dummies(df, columns= ['Sex'], dtype=np.uint8)

    # Insert Missing_Age
    modData.insert(4, "Missing_Age", 0)

    # Loop through and change "NaN" to zero for age
    modData['Age'] = modData['Age'].fillna(0)

    for index, row in modData.iterrows():
        if modData.at[index, 'Age'] == 0:
            modData.at[index, 'Missing_Age'] = 1

    # Create a Missing_Cabin column as well as change NaN in cabin to 0
    modData['Cabin'] = modData['Cabin'].fillna(0)
    modData.insert(9, "Missing_Cabin", 0)

    modData['Cabin'] = modData['Cabin'].apply(convert_cabin)

    for index, row in modData.iterrows():
        if modData.at[index, 'Cabin'] == 0:
            modData.at[index, 'Missing_Cabin'] = 1


    # Remove un-wanted columns        
    modData = modData.drop(columns=['Name', 'Ticket', 'Embarked'])
    
    # Min-max normalize Age and Fare columns
    modData['Age'] = ((modData['Age'] - modData['Age'].min())/(modData['Age'].max() - modData['Age'].min()))
    modData['Fare'] = ((modData['Fare'] - modData['Fare'].min())/(modData['Fare'].max() - modData['Fare'].min()))

    return modData





def main():
    
    cleanTest = clean_data('titantic_data/test.csv')
    cleanTrain = clean_data('titantic_data/train.csv')

    cleanTest.to_csv('titantic_data/cleanTest.csv')
    cleanTrain.to_csv('titantic_data/cleanTrain.csv')
    


























if __name__ == "__main__":
    main()