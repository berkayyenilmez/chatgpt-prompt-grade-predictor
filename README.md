# Project Title

## Overview of the Repository

This repository contains a series of scripts and code pieces designed to [briefly describe the main function]. Below is a list of the main components and their roles in the project:

- `hw_score_predict.ipynb`:
    ```python
    data_path = "dataset/dataset/*.html"
    code2convos = dict()
    total_code_response_list = []
    pbar = tqdm.tqdm(sorted(list(glob(data_path))))
    for path in pbar:
        code_block_count = 0
        file_code = os.path.basename(path).split(".")[0]
        with open(path, "r", encoding="latin1") as fh:
            html_page = fh.read()
            soup = BeautifulSoup(html_page, "html.parser")
    
            data_test_id_pattern = re.compile(r"conversation-turn-[0-9]+")
            conversations = soup.find_all("div", attrs={"data-testid": data_test_id_pattern})
            convo_texts = []
            last_user_text = None  
    
            for i, convo in enumerate(conversations):
                user_div = convo.find("div", attrs={"data-message-author-role": "user"})
                assistant_div = convo.find("div", attrs={"data-message-author-role": "assistant"})
    
                # When a user message is found, save it to last_user_text
                if user_div:
                    last_user_text = user_div.text.strip()
                if assistant_div and assistant_div.find("code"):  
                    code_block_count += 1
    
                # When an assistant message follows a user message, pair them
                if assistant_div and last_user_text is not None:
                    convo_texts.append({
                        "role": "user",
                        "text": last_user_text,
                        "response": assistant_div.text.strip()  
                    })
                    last_user_text = None  # Reset last_user_text after pairing
            total_code_response_list.append((file_code, code_block_count))
            
            code2convos[file_code] = convo_texts
    total_code_response_df = pd.DataFrame(total_code_response_list, columns=['code', 'code_responses'])
    ```
### Explanation: 
The provided Python code block is part of a script that processes HTML files to extract and analyze conversations. It sets a path to the HTML files, initializes a dictionary to store conversations, and a list to track the count of code responses. Using the tqdm library, it creates a progress bar to iterate through each file. As it loops, it opens and reads the file's content, uses BeautifulSoup to parse the HTML, and compiles a list of conversations based on specific attributes using regular expressions. For each conversation turn, it checks for user and assistant messages, counting code blocks within the assistant's responses, and pairs them if they follow a user message. These pairs are then added to the code2convos dictionary with a unique file code as the key. Finally, it compiles a DataFrame from the list of code responses. This structured approach efficiently organizes conversational data for subsequent processing steps in the project.

---

- `hw_score_predict.ipynb`:
    ```python
    prompts = []
    responses = []
    code2prompts = defaultdict(list)
    for code , convos in code2convos.items():
        user_prompts = []
        for conv in convos:
            if conv["role"] == "user":
                prompts.append(conv["text"])
                responses.append(conv["response"])
                user_prompts.append(conv["text"])
        code2prompts[code] = user_prompts
    ```

### Explanation: 
For each html code in dataset this code creates a dictionary entry for them and put the corresponding prompts for that user.

---

- `hw_score_predict.ipynb`:
    ```python
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from spellchecker import SpellChecker
    # Download stopwords set
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    spell = SpellChecker()
    # Define a preprocessing function
    def preprocess_text(text):
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove short words
        text = ' '.join([word for word in text.split() if len(word) > 2])
        # Lemmatization and stop words removal
        text = ' '.join([
            lemmatizer.lemmatize(word) for word in text.split() 
            if word not in stopwords.words('english')
        ])
        # Tokenize the text
        words = text.split()
    
        # Only keep words that are spelled correctly
        words = [word for word in words if word in spell or not spell.unknown([word])]
    
        # Rejoin into a single string
        text = ' '.join(words)
        return text
    
    # Preprocess each prompt and question
    processed_prompts = [preprocess_text(prompt) for prompt in prompts]
    processed_questions = [preprocess_text(question) for question in questions]
    
    # Fit the vectorizer on the processed text with a strict token pattern
    token_pattern = r'\b[a-zA-Z]{2,}\b'  # Only English alphabetic characters
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=token_pattern)
    
    vectorizer.fit(processed_prompts + processed_questions)
    features = vectorizer.get_feature_names_out()
    
    # Transform the questions and convert to DataFrame
    questions_TF_IDF = pd.DataFrame(
        vectorizer.transform(processed_questions).toarray(), 
        columns=features
    )
    
    # Manual filtering of non-English features
    english_feature_columns = [col for col in questions_TF_IDF.columns if re.fullmatch(r'[a-zA-Z]+', col)]
    questions_TF_IDF = questions_TF_IDF[english_feature_columns]
    questions_TF_IDF
    ```

### Explanation: 
For each prompt in each html element and for each question in assignment this code creates a word vector each row corresponds to the question in the assignment and each feature(column) is the unique word in both assignment questions, user prompts.

---
- `hw_score_predict.ipynb`:
    ```python
    code2prompts_tf_idf = dict()
    for code, user_prompts in code2prompts.items():
        if len(user_prompts) == 0:
            # some files have issues
            print(code+".html")
            continue
    
        # Apply the same preprocessing to user_prompts
        processed_user_prompts = [preprocess_text(prompt) for prompt in user_prompts]
    
        # Transform preprocessed prompts into TF-IDF matrix
        prompts_TF_IDF = pd.DataFrame(
            vectorizer.transform(processed_user_prompts).toarray(), 
            columns=vectorizer.get_feature_names_out()
        )
    
        code2prompts_tf_idf[code] = prompts_TF_IDF
    ```

### Explanation: 
This creates a unique entry for each html code that contains the prompts of the user as row and words used in all prompts and questions as column

---
- `hw_score_predict.ipynb`:
    ```python
    code2cosine = dict()
    for code, user_prompts_tf_idf in code2prompts_tf_idf.items():
        code2cosine[code] = pd.DataFrame(cosine_similarity(questions_TF_IDF,user_prompts_tf_idf))
    ```

### Explanation: 
code2cosine dictionary contains the calculated cosine distances for each prompt that html has

---
- `hw_score_predict.ipynb`:
    ```python
    code2questionmapping = dict()
    for code, cosine_scores in code2cosine.items():
        code2questionmapping[code] = code2cosine[code].max(axis=1).tolist()
    
    
    question_mapping_scores = pd.DataFrame(code2questionmapping).T
    question_mapping_scores.reset_index(inplace=True)
    question_mapping_scores.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
    question_mapping_scores.rename(columns={"index" : "code"}, inplace=True)
    
    
    # Define the weights for each question based on their contribution to the overall grade.
    weights = [ 5, 15, 5, 10, 20, 15, 20, 10]
    
    # Assuming 'question_mapping_scores' is your DataFrame and it's already loaded.
    # Apply the weights to each column (excluding the 'code' column).
    for i, weight in enumerate(weights):
        question_mapping_scores[f'Q_{i}'] *= weight
    question_mapping_scores['weighted_total'] = question_mapping_scores[[f'Q_{i}' for i in range(len(weights))]].sum(axis=1)
    question_mapping_scores = question_mapping_scores[['code', 'weighted_total']]
    ```

### Explanation: 
question_mapping_scores is a dataframe where for each prompt the student asked to gpt the cosine similarity is calculated and the maximum is taken so each column represent the maximum similarity for the corresponding question in the assignment considering all the prompts user asked.

---
- `hw_score_predict.ipynb`:
    ```python
    code2features = defaultdict(lambda : defaultdict(int))
    
    keywords2search = ["error", "no", "Entropy"]
    keywords2search = [k.lower() for k in keywords2search]
    
    for code, convs in code2convos.items():
        if len(convs) == 0:
            print(code)
            continue
        for c in convs:
            text = c["text"].lower()
            if c["role"] == "user":

                code2features[code]["#user_prompts"] += 1
                
                for kw in keywords2search:
                    code2features[code][f"#{kw}"] +=  len(re.findall(rf"\b{kw}\b", text))
    
                code2features[code]["prompt_avg_chars"] += len(text)
            else:
                # ChatGPT Responses
                code2features[code]["response_avg_chars"] += len(text)
    
            code2features[code]["prompt_avg_chars"] /= code2features[code]["#user_prompts"]
    ```

### Explanation: 
This code creates a dataframe that has number of occurrences user prompts, error etc.

---
- `hw_score_predict.ipynb`:
    ```python
    # reading the scores
    scores = pd.read_csv("scores.csv", sep=",")
    scores["code"] = scores["code"].apply(lambda x: x.strip())
    
    # selecting the columns we need and we care
    scores = scores[["code", "grade"]]
    std_dev = scores["grade"].std()
    mean = scores["grade"].mean()
    ```

### Explanation: 
Reads the scores of the data

---
- `hw_score_predict.ipynb`:
    ```python
    temp_df = pd.merge(df, scores, on='code', how="left")
    temp_df = pd.merge(temp_df, total_code_response_df, on='code', how='left')
    temp_df.dropna(inplace=True)
    temp_df.drop_duplicates("code",inplace=True, keep="first")
    ```

### Explanation: 
temp_df merged by the vector matrix and the features we have added. The total_code_response_df stores the amount of responses by chatgpt that has codeblock, thus it has the following features; code	#user_prompts	#error	#no	#entropy	prompt_avg_chars	weighted_total	code_responses

---
- `hw_score_predict.ipynb`:
    ```python
    regressor = DecisionTreeRegressor(random_state=0,criterion='squared_error', max_depth=10)
    regressor.fit(X_train, y_train)
    # Prediction
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    
    # Calculation of Mean Squared Error (MSE)
    print("MSE Train:", mean_squared_error(y_train,y_train_pred))
    print("MSE TEST:", mean_squared_error(y_test,y_test_pred))
    
    print("R2 Train:", r2_score(y_train,y_train_pred))
    print("R2 TEST:", r2_score(y_test,y_test_pred))
    ```

### Explanation: 
This is the first model in our code, basic decision tree prediction based on the data in the temp_df and scores

---
- `hw_score_predict.ipynb`:
    ```python
    import re
    import string
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    
    
    nltk.download('punkt')
    nltk.download('stopwords')
    
    
    stemmer = SnowballStemmer("english")
    
    stop_words = set(stopwords.words('english'))
    print(scores)
    # Assuming 'scores' and 'code2convos' are defined and available
    subset = scores[scores["grade"] >= 98]
    code_values = subset["code"]
    vocab_count = {}
    
    def join_words(words):
        return ' '.join(words)
    
    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]
    
    def stem_words(words):
        return [stemmer.stem(word) for word in words]
    
    def preprocess_text(text):
        text = re.sub(r'[{}0-9]+'.format(re.escape(string.punctuation)), '', text.lower())
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        return tokens
    
    for file_code in code_values:
        for chat in code2convos[file_code]:
            processed_tokens = preprocess_text(chat['response'])
            processed_tokens = remove_stopwords(processed_tokens)
            processed_tokens = stem_words(processed_tokens)
            chat['response'] = join_words(processed_tokens)
    
            for word in processed_tokens:
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 1
    
    # Sort and display the word count
    sorted_vocab = sorted(vocab_count.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_vocab:
        print(f"{word}: {count}")
    len(sorted_vocab)
    ```

### Explanation: 
This code block creates a response word vector based on the responses of histories that is graded higher than 97 and creates a vocabulary that contains unique words as keys and number of occurrence of that word as key.

---
- `hw_score_predict.ipynb`:
    ```python
    flattened_data = []
    for code, convos in code2convos.items():
        for convo in convos:
            # Add the code as a key in each dictionary
            convo_with_code = convo.copy()
            convo_with_code['code'] = code
            flattened_data.append(convo_with_code)
    
    
    df = pd.DataFrame(flattened_data)
    
    # If you want to reorder the columns to have 'code' as the first column
    df = df[['code', 'response']]  # Add other column names as needed
    
    # Now df is your desired DataFrame
    bagofwords_data = pd.merge(scores, df, on="code", how="left")
    grouped_data = bagofwords_data.groupby('code').agg({'response': lambda x: list(x)}).reset_index()
    
    # If you want to keep the 'grade' column, you need to decide how to handle multiple grades per code.
    # For example, you can take the mean, max, or min. Here's an example using mean:
    grouped_data = bagofwords_data.groupby('code').agg({'grade': 'mean', 'response': lambda x: list(x)}).reset_index()
    
    # Now 'grouped_data' is your desired DataFrame
    # grouped_data
    grouped_data['response'] = grouped_data['response'].apply(lambda responses: ' '.join(str(response) for response in responses))
    codes_to_remove = ["139235c7-736c-4237-92f0-92e8c116832c", 
                       "668ad17e-0240-49f7-b5a7-d22e502554c6", 
                       "b0640e51-6879-40cb-a4f5-329f952ef99d", 
                       "da6b70d5-29f6-491a-ad46-037c77067128"]
    
    # Remove rows with these codes
    grouped_data = grouped_data[~grouped_data['code'].isin(codes_to_remove)]
    filtered_grouped_data = grouped_data[grouped_data['code'].isin(temp_df['code'])]
    
    
    # filtered_grouped_data = pd.merge(filtered_grouped_data, total_code_response_df, on='code', how='left')
    # filtered_grouped_data = pd.merge(filtered_grouped_data, question_mapping_scores, on='code', how='left')
    
    temp_df2 = temp_df.drop('grade', axis=1)
    filtered_grouped_data = pd.merge(filtered_grouped_data, temp_df2, on="code", how="left")
    
    X = filtered_grouped_data[['response', 'weighted_total', 'code_responses', 'prompt_avg_chars', '#user_prompts']]
    y = filtered_grouped_data['grade']
    
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

### Explanation: 
This code block merges the previous temp_df and the responses that the user receieved from chatgpt then creates training data and test data based on the merged dataframe.

---
- `hw_score_predict.ipynb`:
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import Adam
    import numpy as np
    from scipy.sparse import csr_matrix, isspmatrix
    from sklearn.preprocessing import MinMaxScaler
    # Get the top 300 words
    vocab_list = [word for word, count in sorted_vocab[:500]]
    
    # Initialize CountVectorizer with the top 300 words as the vocabulary
    vectorizer = CountVectorizer(vocabulary=vocab_list, max_features=500, preprocessor=lambda x: x)
    # vectorizer.fit_transform(train_data['response'])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform the training labels
    train_labels_normalized = scaler.fit_transform(train_labels.values.reshape(-1, 1))
    test_labels_normalized = scaler.fit_transform(test_labels.values.reshape(-1, 1))
    # Vectorize the responses
    train_vectors = vectorizer.transform(train_data['response'])
    test_vectors = vectorizer.transform(test_data['response'])
    
    # Convert the BoW vectors to a DataFrame
    train_bow_df = pd.DataFrame(train_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    test_bow_df = pd.DataFrame(test_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Reset the index if needed
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    # Concatenate the BoW DataFrame with the existing features, excluding the original 'response' column
    train_data_combined = pd.concat([train_bow_df, train_data.drop('response', axis=1)], axis=1)
    test_data_combined = pd.concat([test_bow_df, test_data.drop('response', axis=1)], axis=1)
     
    train_data_combined_np = train_data_combined.to_numpy()
    test_data_combined_np = test_data_combined.to_numpy()
    
    # Fit the model using the NumPy arrays
    
    model = Sequential([
        Dense(units=64, activation='relu', input_shape=[train_vectors.shape[1]+4]),
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dropout(0.2),
        Dense(units=1, activation= 'sigmoid')  # Output layer for regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(train_data_combined_np, train_labels_normalized, epochs=10)
    
    # Evaluate the model
    predictions = model.predict(test_data_combined_np)
    predictions_rescaled = scaler.inverse_transform(predictions)
    mse = mean_squared_error(test_labels, predictions_rescaled)
    rmse = np.sqrt(mse)
    print(train_labels)
    print(predictions)
    print(f"RMSE: {rmse}")
    ```

### Explanation: 
The vocabulary we created based on the responses of histories that is graded by 98 or more is used to train the BagOfWords vectorizer. Then the train and test data is tranformed based on the results of the BagOfWords vectorizer which is trained by top 500 occurring words. Then we merge the created feature by this transform with temp_df and feed it to the neural network model

---
- `hw_score_predict.ipynb`:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import KFold
    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    rf_regressor.fit(train_data_combined_np, train_labels_normalized.ravel())
    
    # Make predictions
    rf_predictions = rf_regressor.predict(test_data_combined_np)
    rf_predictions_rescaled = scaler.inverse_transform(rf_predictions.reshape(-1, 1))
    
    # Evaluate the model
    rf_mse = mean_squared_error(test_labels, rf_predictions_rescaled)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(test_labels, rf_predictions_rescaled)
    
    print(f"Random Forest RMSE: {rf_rmse}")
    print(f"Random Forest R2 Score: {rf_r2}")
    ```

### Explanation: 
We try different model (randomforest) to evaluate.

---
- `hw_score_predict.ipynb`:
    ```python
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Initialize RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=150)
    
    # Setup KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store the metrics for each fold
    rmses = []
    r2_scores = []
    combined_data_np = np.concatenate((train_data_combined_np, test_data_combined_np), axis=0)
    
    y = filtered_grouped_data['grade'].to_numpy().reshape(-1, 1)
    
    # Concatenate with combined_data_np along axis 1 (columns)
    combined_data_with_target_np = np.concatenate((combined_data_np, y), axis=1)
    
    X = combined_data_with_target_np[:, :-1]
    y = combined_data_with_target_np[:, -1]
    
    for train_index, test_index in kf.split(X):
        # Splitting the data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Fit the model on training data
        rf_regressor.fit(X_train, y_train)
    
        # Make predictions on the test set
        rf_predictions = rf_regressor.predict(X_test)
        
        # Evaluate the model
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_rmse = np.sqrt(rf_mse)
       
    
        # Store the metrics
        rmses.append(rf_rmse)
       
    
    # Calculate the average metrics across all folds
    average_rmse = np.mean(rmses)
    print(rmses)
    print(f"Average Random Forest RMSE across all folds: {average_rmse}")
    ```

### Explanation: 
Here we are trying to see if the performance depends on the split or not. To see this we are using k-fold with n_split 5 and we can see how diffrent combination of splits affects the result.

---
- `hw_score_predict.ipynb`:
    ```python
    codes_to_keep = set(temp_df['code'])
    
    code2convos2 = {k: code2convos2[k] for k in code2convos2 if k in codes_to_keep}
    
    code_responses = []
    
    for code , convos in code2convos2.items():
        for conv in convos:
            code_responses.append(conv)
    print(len(code_responses))
    ```

### Explanation: 
Here we take the responses in gpt histories that contains specifically the code block and append it to code_responses array.

---
- `hw_score_predict.ipynb`:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import text
    
    
    original_stop_words = text.ENGLISH_STOP_WORDS
    
   
    combined_stop_words = list(original_stop_words)
    
    
    token_pattern = r'(?u)\b[A-Za-z]{2,}\b'
    
    vectorizer = TfidfVectorizer(
        lowercase=True,                  # Convert all characters to lowercase
        stop_words=combined_stop_words,  # Combined stop words
        token_pattern=token_pattern,     # Custom token pattern for English letters only
    )
    
    
    vectorizer.fit(answer_key + code_responses)
    features = vectorizer.get_feature_names_out()
    
    
    answer_key_TF_IDF = pd.DataFrame(vectorizer.transform(answer_key).toarray(),   columns=vectorizer.get_feature_names_out())
    ```

### Explanation: 
We create a vector matrix out of answer_key and code_responses of the users in answer_key_TF_IDF. Each row corresponds to the question in answer_key which is based on the code answers of a history graded 100 and each column is a word (feature) in all the code responses of chat histories.

---
- `hw_score_predict.ipynb`:
    ```python
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from spellchecker import SpellChecker
    
    # Initialize lemmatizer and spell checker
    lemmatizer = WordNetLemmatizer()
    spell = SpellChecker()
    nltk_stopwords = set(stopwords.words('english'))
    
    def preprocess_text2(text):
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Address concatenated true/false values
        text = re.sub(r'(truefalse|falsetrue)', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove short words
        text = ' '.join([word for word in text.split() if len(word) > 2])
        # Lemmatization and stop words removal
        text = ' '.join([
            lemmatizer.lemmatize(word) for word in text.split() 
            if word not in nltk_stopwords
        ])
        # Tokenize the text
        words = text.split()
    
        # Only keep words that are spelled correctly
        words = [word for word in words if spell.known([word])]
    
        # Rejoin into a single string
        text = ' '.join(words)
        return text
    
    # Preprocess each prompt and question
    # Assuming code_responses and answer_key are lists of text strings that you want to preprocess
    processed_code_responses = [preprocess_text2(response) for response in code_responses]
    processed_answer_key = [preprocess_text2(response) for response in answer_key]
    
    
    # Fit the vectorizer on the processed text with a strict token pattern
    token_pattern = r'\b[a-zA-Z]{2,}\b'  # Only English alphabetic characters
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=token_pattern)
    
    vectorizer.fit(processed_code_responses + processed_answer_key)
    features = vectorizer.get_feature_names_out()
    print(features)
    # Transform the questions and convert to DataFrame
    answer_key_TF_IDF = pd.DataFrame(
        vectorizer.transform(processed_answer_key).toarray(), 
        columns=features
    )
    
    # Manual filtering of non-English features
    english_feature_columns = [col for col in answer_key_TF_IDF.columns if re.fullmatch(r'[a-zA-Z]+', col)]
    answer_key_TF_IDF = answer_key_TF_IDF[english_feature_columns]
    ```

### Explanation: 
Here we preprocess the data by eliminating non-english, numerical etc. words.

---
- `hw_score_predict.ipynb`:
    ```python
    code2answers_tf_idf = dict()
    
    for code, user_prompts in code2convos2.items():
        if len(user_prompts) == 0:
            # some files have issues
            print(code+".html")
            continue
        
        # Apply the same preprocessing to user_prompts
        processed_user_answers = [preprocess_text2(prompt) for prompt in user_prompts]
    
        # Transform preprocessed prompts into TF-IDF matrix
        prompts_TF_IDF = pd.DataFrame(
            vectorizer.transform(processed_user_answers).toarray(), 
            columns=vectorizer.get_feature_names_out()
        )
    
        code2answers_tf_idf[code] = prompts_TF_IDF
    ```

### Explanation: 
aciklama

---
- `hw_score_predict.ipynb`:
    ```python
    code2responsemapping = dict()
    for code, cosine_scores in code2cosine.items():
        code2responsemapping[code] = code2cosine[code].max(axis=1).tolist()
    
    
    response_mapping_scores = pd.DataFrame(code2responsemapping).T
    response_mapping_scores.reset_index(inplace=True)
    response_mapping_scores.rename(columns={i: f"Q_{i}" for i in range(len(questions))}, inplace=True)
    response_mapping_scores.rename(columns={"index" : "code"}, inplace=True)
    ```

### Explanation: 
response_mapping_scores is a dataframe that has each history as a row code and corresponding maximum cosine distance to the question in the column (questions in answer_key).

---
- `hw_score_predict.ipynb`:
    ```python
    codes_in_response_mapping = set(response_mapping_scores['code'])
    
    # Create a boolean mask for rows in temp_df where 'code' is in the codes of response_mapping_scores
    mask = temp_df['code'].isin(codes_in_response_mapping)
    
    # Filter temp_df using the mask
    filtered_temp_df = temp_df[mask]
    grades = filtered_temp_df["grade"].to_numpy()
    response_mapping_scores2 = response_mapping_scores.drop('code', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(response_mapping_scores2, grades, test_size=0.2, random_state=42)
    ```

### Explanation: 
Here we create test and train data out of the mapping scores and grades

---
- `hw_score_predict.ipynb`:
    ```python
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Assuming X_train and X_test are your feature DataFrames
    # Convert column names to strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    
    # Initialize the RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model on the training data
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions on the test data
    rf_predictions = rf_regressor.predict(X_test)
    
    # Evaluate the model
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    print(f"Random Forest RMSE: {rf_rmse}")
    print(f"Random Forest R2 Score: {rf_r2}")
    ```

### Explanation: 
Using randomforest model to predict grades.

---

## Clustering Analysis

### KMeans Clustering

This section applies KMeans clustering to identify the optimal number of clusters in the dataset.


- `hw_score_predict_clustering.ipynb`:

    ```python
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    def find_optimal_clusters(data, max_clusters=10):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_assignments = kmeans.fit_predict(data_scaled)
            silhouette_avg = silhouette_score(data_scaled, cluster_assignments)
            silhouette_scores.append(silhouette_avg)
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        return optimal_clusters, silhouette_scores, cluster_assignments

    X = temp_df[temp_df.columns[1:]].to_numpy()
    optimal_clusters, silhouette_scores, cluster_assignments = find_optimal_clusters(X, max_clusters=10)

    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.show()

    print(f'Optimal Number of Clusters: {optimal_clusters}')
    ```

#### Explanation:
The script finds the best number of clusters for KMeans by comparing silhouette scores.

---
- `hw_score_predict_clustering.ipynb`:

    ```python
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # using the merged df from above
    X = temp_df[temp_df.columns[1:]].to_numpy()

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # optimal number of cluster
    k = 2

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_assignments = kmeans.fit_predict(X_scaled)

    # Visualize clusters using scatter plot
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_assignments, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    ```

#### Explanation:
The script visualizes K-Means clustering results on scaled data, using a scatter plot to depict the clusters formed for two features.

---

- `hw_score_predict_clustering.ipynb`:

    ```python
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Choosing the second most optimal
    k = 4

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_assignments = kmeans.fit_predict(X_scaled)

    # Visualize clusters using scatter plot
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_assignments, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    ```

#### Explanation:
The script applies K-Means clustering with four clusters to the scaled dataset and visualizes the results using a scatter plot, showcasing the cluster distribution across two features.

---

### DBScan Clustering

This section applies DBSCAN clustering to identify dense clusters in the dataset, emphasizing the discovery of non-linearly separable clusters.


- `hw_score_predict_clustering.ipynb`:

    ```python
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Extracting the same features as above
    X = temp_df[temp_df.columns[1:]].to_numpy()

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    cluster_assignments = dbscan.fit_predict(X_scaled)

    # Visualize clusters
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_assignments, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    ```

#### Explanation:
The script applies DBSCAN clustering to the scaled dataset and visualizes the results. DBSCAN, a density-based clustering method, groups together closely packed points, marking outliers as noise. The scatter plot shows how data points are clustered or marked as outliers.

---



The project is structured to [explain how the different scripts and code pieces are linked, e.g., data flow or execution order].

## Methodology

Here we provide a high-level explanation of the methodology employed in the project. This section covers the theoretical basis, the algorithms or models used, and any significant reasons for choosing specific approaches. It also outlines the solutions offered by the project, addressing the problems or challenges the project is intended to solve.

## Results

Our experiments yielded the following key findings:

- **Finding 1**: A brief explanation supported by visuals (if applicable).
- **Finding 2**: Summary of the result and its implications.

Figures and tables illustrating our results are included below:

![Figure 1 Description](path/to/figure1.png)
*Figure 1: Caption describing the content and significance of the figure.*

| Table 1        | Column 1       | Column 2       |
|----------------|----------------|----------------|
| Row 1          | Data 1         | Data 2         |
| Row 2          | Data 3         | Data 4         |
*Table 1: Caption explaining what the table shows.*

## Team Contributions

- **Team Member 1**: Detailed contributions.
- **Team Member 2**: Detailed contributions.
- [Additional team members and their contributions]

(Replace the placeholders with the actual names and contributions of the team members.)

## Additional Sections

You may also include additional sections such as 'Installation', 'Usage', 'Dependencies', 'Contributing', 'License', and 'Acknowledgments' as needed to provide more context and information about your project.

