# NLP--HookahTweets

Abstract:

CTSI wants to understand the public attitude and user of hookah through analyzing Tweets until 2023 to provide more insights about Hookah regulation and prevention campaigns. Thus, the goal of this project is to predict the attitude (positive, negative, neutral) and user or not towards hookah for a given Tweet, and analyze any insightful trends and patterns of the prediction. To deliver against this goal, the team employed various large language model classification algorithms, e.g., RoBERTa and Llama 2, to successfully classify and predict the attitude and user to hookah of a given Tweet. Additionally, the team leveraged data augmentation and hyperparameter tuning to mitigate small dataset problems and class imbalance performance issues. Finally, the team delivered various visualizations of commercial and non-commercial Tweets, and the geographical and longitudinal distribution of attitude and users through Python and Tableau.

Installation:

	•	Code is stored in two Google Colaboratory notebooks. There are no known version requirements.
Usage:
	•	Code: Text_classification_using_roberta.ipynb:
	•	Data: augmented_train_data.csv:
                      random_2300.csv
	          test_data.xlsx
	•	Code: fine_tune_llama_2_for_hookah_tweet.ipynb:
	•	Data: augmented_train_data.csv:
                      Train_2300.csv
Code Details: Text_classification_using_roberta.ipynb
Importing Python Libraries and preparing the environment
This section imports the libraries and modules needed to run the script Libraries are:

Preparing the Dataset and Dataloader
This section creates the Dataset class, which defines how the text is pre-processed, and uses the Dataloader feed the data in batches to the model.

Creating the Neural Network for Fine Tuning
This section defines the structure and parameters of the model.

Fine Tuning the Model
This section define a training function and some key parameters for training the model

Validating the Model
This section validates the performance of the model through accuracy and F1 score.

Code Details: fine_tune_llama_2_for_hookah_tweet.ipynb
Installation and Import Packages
This section installs the packages to run our notebook.

Load Data and Preprocess
This section loads data from csv file and preprocess data through generating prompt for future training

Parameter
This section set the parameters for the model
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8, # 4
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=5e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=1024,
    neftune_noise_alpha=10,
)

Performance and Results
This section uses tensorboard and classification report to show the performance and results of model
	•	Accuracy
	•	F1 Score
	•	Precision
	•	Recall Rate
