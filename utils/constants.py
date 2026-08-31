INPUT_CONTEXT_KEYWORD = "###INPUT_CONTEXT###"

# Constants for datetime parsing
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
]

# TimerBed Output Formats
TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT = """
Please respond with the following format:

Answer Choice: [Your Answer Choice Here]
Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1]

Do not deviate from the above format."""

# OFFICIAL OUTPUT FORMATS
# MTBench Trend Classification (Section 4.2)
MTBENCH_TREND_LABELS = ["<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"]

# MTBench Correlation (Section 4.4)
MTBENCH_CORRELATION_LABELS = [
    "Strong Positive Correlation",
    "Moderate Positive Correlation",
    "No Correlation",
    "Moderate Negative Correlation",
    "Strong Negative Correlation",
]

# MTBench MCQA
MTBENCH_MCQA_OPTIONS = ["A", "B", "C", "D"]


# TASK CONFIGURATIONS
TASK_CONFIGS = {
    # MTBench Finance Tasks (Section 4)
    "mtbench_finance_trend": {
        "task_type": "classification",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": MTBENCH_TREND_LABELS,
        "output_format": "^^^label^^^",  # Official: wrap in ^^^
        # OFFICIAL: MTBench finance/meta_prompt.py finance_classification_metaprompt_generation()
        "task_instruction": """You are a financial prediction expert with knowledge of advanced machine learning models and time-series analysis. 
Your goal is to predict the stock trend (rise, neutral, or fall) based on any given inputs. The inputs may include time series, news, or other relevant data.
The stock prices recorded at 1-hour intervals over the last month from {start_datetime} to {end_datetime}.

###INPUT_CONTEXT###

Task: Analyze the provided data to identify future trends of the stock performance.
Output: Provide a prediction for the stock trend categorized one of the following labels:
- "<-4%"
- "-2% ~ -4%"
- "-2% ~ +2%"
- "+2% ~ +4%"
- ">+4%"

Please briefly explain how you reached the prediction.
Then, wrap your final answer in: ^^^label^^^""",
    },
    "mtbench_finance_forecasting": {
        "task_type": "regression",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": None,
        "output_format": "Predicted Prices: val1, val2, ...",
        # OFFICIAL: MTBench finance/meta_prompt.py finance_mse_metaprompt_generation()
        "task_instruction": """You are an AI assistant trained in data analysis and modeling. Your task is to conduct a research-based timeseries estimation for the next {prediction_length} time steps based on provided historical price movements and/or related news articles. This analysis aims to explore patterns in the given dataset and should not be considered financial advice.
The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity.

###INPUT_CONTEXT###

Please return your estimated values in a structured format as a list of float numbers.
Ensure the output (exactly {prediction_length} values) follows this format strictly: Predicted Prices: value1, value2, ..., valueN.""",
    },
    # MTBench Finance Indicator Tasks - MACD and Bollinger Bands are distinct tasks
    "mtbench_finance_indicator_macd": {
        "task_type": "regression",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": None,
        # OFFICIAL: MTBench uses "Predicted Prices:" even for MACD (see meta_prompt.py line 112)
        "output_format": "Predicted MACD: val1, val2, ...",
        # OFFICIAL: MTBench finance/meta_prompt.py finance_macd_metaprompt_generation()
        "task_instruction": """You are an AI assistant trained in data analysis and modeling. Your task is to Predict the future Moving Average Convergence Divergence (MACD) values for the next {prediction_length} time steps based on provided historical timeseries movements and/or related news articles. The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity.

###INPUT_CONTEXT###

Please return your predicted MACD values in a structured format as a list of float numbers. Please predict the real possible values, do not use the naive linear extrapolation or similar methods.
Ensure the output (exactly {prediction_length} values) follows this format strictly: Predicted MACD: value1, value2, ..., valueN.""",
    },
    "mtbench_finance_indicator_bb": {
        "task_type": "regression",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": None,
        # OFFICIAL: MTBench uses "Predicted MACD:" for BB too (see meta_prompt.py line 161)
        "output_format": "Predicted BB: val1, val2, ...",
        # OFFICIAL: MTBench finance/meta_prompt.py finance_bb_metaprompt_generation()
        "task_instruction": """You are an AI assistant trained in data analysis and modeling. Your task is to Predict the future upper Bollinger Band (BB) values for the next {prediction_length} time steps based on provided historical price movements and/or related news articles. The input time series spans from {start_datetime} to {end_datetime}, with a granularity of {granularity}. The estimation period extends from {end_datetime} to {pred_end_datetime}, maintaining the same granularity.

###INPUT_CONTEXT###

Please return your estimated upper Bollinger Band (BB) values in a structured format as a list of float numbers.
Ensure the output (exactly {prediction_length} values) follows this format strictly: Predicted BB: value1, value2, ..., valueN.""",
    },
    "mtbench_finance_correlation": {
        "task_type": "classification",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": MTBENCH_CORRELATION_LABELS,
        "output_format": "label_only",
        # OFFICIAL: MTBench finance/meta_prompt.py finance_correlation_metaprompt_generation()
        "task_instruction": """You are an expert in finance and stock market analysis. Based on the given 30-day historical stock price time series and/or a financial analysis published at the last timestamp of the time series, your task is to predict the correlation between the stock's price fluctuations in the next 7 days and the analysis sentiment (positive correlation indicates that positive analysis leads to price increase and negative analysis leads to price decrease).
Take into account external factors or market conditions that might affect stock price movement. The stock price of {sticker} is between {start_datetime} to {end_datetime}, time interval is 1 hour. News published at {news_timestamp}.

###INPUT_CONTEXT###

Return your answer in one of the following without any other words: Strong Positive Correlation, Moderate Positive Correlation, No Correlation, Moderate Negative Correlation, Strong Negative Correlation.
Answer:""",
    },
    "mtbench_finance_mcqa": {
        "task_type": "mcqa",
        "benchmark": "MTBench",
        "domain": "finance",
        "labels": MTBENCH_MCQA_OPTIONS,
        "output_format": "letter_only",
        # OFFICIAL: MTBench finance/meta_prompt.py finance_mcqa_metaprompt_generation()
        "task_instruction": """You are an expert in finance and stock market analysis. Your task is to answer the question based on the given 30-day historical stock price time series and/or a financial analysis published at the last timestamp of the time series. Return your answer only in the letter (A, B, C, or D). 
The stock price of {sticker} is between {start_datetime} to {end_datetime}, time interval is 1 hour. News published at {news_timestamp}.

###INPUT_CONTEXT###

Answer:""",
    },
    # MTBench Weather Tasks
    "mtbench_weather_forecasting": {
        "task_type": "regression",
        "benchmark": "MTBench",
        "domain": "weather",
        "labels": None,
        "output_format": "Predicted Temperatures: val1, val2, ...",  # OFFICIAL format!
        # OFFICIAL: MTBench weather/meta_prompt.py temperature_forecast_metaprompt_generation()
        "task_instruction": """You are a weather forecasting AI. Your task is to predict the next {prediction_length} time steps for temperature based on the given data. The input time series represents temperature readings from {start_datetime} to {end_datetime}, with a granularity of {granularity}. The prediction should cover the period for next {next_days} days with the same granularity ({granularity}). 
This data is from a location in the United States, where summers are hot and winters are cold. Weather conditions can also be affected by storms, heavy rain, and cold fronts. The daytime is usually warmer than the nighttime. Every 24 temperature readings represent a full day from 00:00 to 23:00.

###INPUT_CONTEXT###

Return your prediction (exactly {prediction_length} values, rounded to 2 decimal places) as a list of float values in plain text, strictly following this format: Predicted Temperatures: value1, value2, ..., valueN.
Ensure no extra text or explanations are included.""",
    },
    "mtbench_weather_trend": {
        "task_type": "classification",
        "benchmark": "MTBench",
        "domain": "weather",
        "labels": ["increasing", "decreasing", "stable"],
        "output_format": "single_word",  # OFFICIAL: just one word!
        # OFFICIAL: MTBench weather/meta_prompt.py temperature_trend_metaprompt_generation()
        "task_instruction": """You are a weather forecasting AI. Your task is to analyze the past {past_days} days's of temperature trend and predict the temperature trend for the next {next_days} days'. The input time series represents temperature readings from {start_datetime} to {end_datetime}, with a granularity of {granularity}.
This data is from a location in the United States, where summers are hot and winters are cold. Weather conditions can also be affected by storms, heavy rain, and cold fronts. The daytime is usually warmer than the nighttime.Every 24 temperature readings represent a full day from 00:00 to 23:00.

###INPUT_CONTEXT###

Based on the information you received, predict the temperature trend for the next {next_days} days. Calculate the mean temperature of the last 24-hour period (i.e., the most recent day in the input) and compare it with the mean temperature of the first predicted day. If the difference is greater or equal than 0.5, classify the trend as 'increasing'. If the difference is less or equal than -0.5, classify the trend as 'decreasing'. Otherwise, classify it as 'stable'.
Return one word in 'increasing', 'decreasing', or 'stable', without any reasoning text or extra words.""",
    },
    # MTBench Weather Indicator Task (only MACD variant exists for weather)
    "mtbench_weather_indicator_macd": {
        "task_type": "regression",
        "benchmark": "MTBench",
        "domain": "weather",
        "labels": None,
        # OFFICIAL: MTBench weather indicator returns 3 values (see meta_prompt.py line 197)
        "output_format": "Highest temperature: X, Lowest temperature: Y, Temperature difference: Z",
        # OFFICIAL: MTBench weather/meta_prompt.py temperature_indicator_metaprompt_generation()
        "task_instruction": """You are a weather forecasting AI. Your task is to analyze the past {past_days} days's of temperature trend and predict the next {next_days} days's highest temperature and lowest temperature as well as the temperature difference between the highest and lowest temperature based on the given data.
The input time series represents temperature readings from {start_datetime} to {end_datetime}, with a granularity of {granularity}. This data is from a location in the United States, where summers are hot and winters are cold. Weather conditions can also be affected by storms, heavy rain, and cold fronts. The daytime is usually warmer than the nighttime. Every 24 temperature readings represent a full day from 00:00 to 23:00.

###INPUT_CONTEXT###

Now, you need to predict the next {next_days} days's highest temperature, lowest temperature and temperature difference between the highest and lowest temperature based on the data provided.
Your response should be in the format:'Highest temperature: X, Lowest temperature: Y, Temperature difference: Z' without extra analysis and other words.""",
    },
    "mtbench_weather_mcqa": {
        "task_type": "mcqa",
        "benchmark": "MTBench",
        "domain": "weather",
        "labels": MTBENCH_MCQA_OPTIONS,
        "output_format": "letter_only",
        # OFFICIAL: MTBench weather/meta_prompt.py weather_mcqa_metaprompt_generation()
        "task_instruction": """You have a {past_days}-day temperature time series, a weather event report published on the last day of time series. Answer the question, return your answer in single letter (A, B, C, D) without other words.
The {past_days}-day temperature time series is between {start_datetime} to {end_datetime}, time interval is 1 hour.

###INPUT_CONTEXT###

Answer:""",
    },
    # TimerBed Classification Tasks (6 datasets)
    # labels from: benchmarks/TimerBed/LLMs/Dataset/ and timerbed_loader.py CLASSES
    "timerbed_har": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "HAR",
        # labels from VL-Time paper (NAACL 2025) Section F.1
        "labels": ["walking", "walking upstairs", "walking downstairs", "sitting", "standing", "laying down"],
        # TimerBed uses Answer Choice + Confidence Score format (see LLMs/Method/prompt.py)
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1
        "task_instruction": """As a human activity recognition expert, you are tasked with determining the type of activity performed by the subject based on the accelerometer record series along the x, y, and z axes over time.

###INPUT_CONTEXT###

Question: What activity is represented by this accelerometer sensor data?
Choices: ["walking", "walking upstairs", "walking downstairs", "sitting", "standing", "laying down"]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    "timerbed_ecg": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "ECG",
        # labels from VL-Time paper (NAACL 2025) Section F.1
        "labels": [
            "normal sinus rhythm",
            "fibrillation",
            "alternative rhythm",
            "too noisy to be classified",
        ],
        # TimerBed uses Answer Choice + Confidence Score format
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1
        "task_instruction": """As a cardiologist, you are tasked with classifying a patient's heart condition based on single-lead ECG recordings.

###INPUT_CONTEXT###

Question: What is the cardiac rhythm type from this ECG heartbeat signal?
Choices: ["normal sinus rhythm", "fibrillation", "alternative rhythm", "too noisy to be classified"]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    "timerbed_ctu": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "CTU",
        # labels from VL-Time paper (NAACL 2025) Section F.1 - CTU is Computers dataset (desktop/laptop)
        "labels": ["desktop", "laptop"],
        # TimerBed uses Answer Choice + Confidence Score format
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1
        "task_instruction": """Play as a computer energy consumption analysis expert, please correctly determine whether this computer is a desktop or a laptop based on the 24-hour power consumption data.

###INPUT_CONTEXT###

Question: What is the type of this computer?
Choices: ["desktop", "laptop"]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    "timerbed_emg": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "EMG",
        # labels from benchmarks/TimerBed/LLMs/Dataset/EMG/demo.csv
        "labels": ["healthy", "suffering from neuropathy", "suffering from myopathy"],
        # TimerBed uses Answer Choice + Confidence Score format
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1
        "task_instruction": """As an Electromyograms (EMG) analysis expert, you are tasked with determining the type of the subject based on the EMG record.

###INPUT_CONTEXT###

Question: What is the type of this subject?
Choices: ["healthy", "suffering from neuropathy", "suffering from myopathy"]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    "timerbed_rcw": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "RCW",
        # OFFICIAL labels from benchmarks/TimerBed/LLMs/Dataset/RCW/demo.csv
        "labels": ["There is no right whale call in the image.", "There is a right whale call in the image."],
        # OFFICIAL: TimerBed uses Answer Choice + Confidence Score format
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1
        "task_instruction": """Play the role of a marine biology expert: is there a right whale call in the record?

###INPUT_CONTEXT###

Question: Is there a right whale call in the record?
Choices: ["There is no right whale call in the image.", "There is a right whale call in the image."]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    "timerbed_tee": {
        "task_type": "classification",
        "benchmark": "TimerBed",
        "domain": "TEE",
        # labels from VL-Time paper (NAACL 2025) Section F.1 - TEE is Lightning7 dataset
        "labels": [
            "CG Positive Initial Return Stroke",
            "IR Negative Initial Return Stroke",
            "SR Subsequent Negative Return Stroke",
            "I Impulsive Event",
            "I2 Impulsive Event Pair",
            "KM Gradual Intra-Cloud Stroke",
            "O Off-record",
        ],
        # TimerBed uses Answer Choice + Confidence Score format
        "output_format": "answer_choice_confidence",
        # OFFICIAL: VL-Time paper (NAACL 2025) Section E.1 - Full description from timerbed_loader.py
        "task_instruction": """Based on the power density time series data and select the transient electromagnetic event that best matches. The FORTE satellite detects transient electromagnetic events associated with lightning using a suite of optical and radio-frequency (RF) instruments. There are 7 event types. CG Positive Initial Return Stroke: A positive charge is lowered from a cloud to the ground. The characteristic feature of this type of event in the power density time series is a sharp turn-on of radiation, followed by a few hundreds of microseconds of noise; IR Negative Initial Return Stroke: A negative charge is lowered from a cloud to ground. The power waveform slowly ramps up to a level known as an attachment point, where a large surge current causes the VHF power to 'spike'. This attachment is followed by an exponentially shaped decline in the waveform.; SR Subsequent Negative Return Stroke: A negative charge is lowered from a cloud to ground. As the name implies, subsequent return strokes come after initial return strokes. Note that subsequent positive return strokes don't exist. I Impulsive Event: Typically an intra-cloud event characterized by a sudden peak in the waveform. I2 Impulsive Event Pair: Another intra-cloud event characterized by sudden peaks in the waveform that come in closely separated pairs. These are also called TIPPs (Trans-Ionospheric Pulse Pairs). KM Gradual Intra-Cloud Stroke: An intra-cloud event which increases in power more gradually than an impulsive event. O Off-record: 800 microseconds was not enough to fully capture the lightning event.

###INPUT_CONTEXT###

Question: What is the type of this transient electromagnetic event?
Choices: ["CG Positive Initial Return Stroke", "IR Negative Initial Return Stroke", "SR Subsequent Negative Return Stroke", "I Impulsive Event", "I2 Impulsive Event Pair", "KM Gradual Intra-Cloud Stroke", "O Off-record"]"""
        + TIMERBED_ANSWER_CHOICE_CONFIDENCE_FORMAT,
    },
    # TSQA Tasks
    # OFFICIAL: TSQA answers are full sentences (see benchmarks/TSQA/*.csv)
    "tsqa_classification": {
        "task_type": "classification",
        "benchmark": "TSQA",
        "domain": None,
        "labels": None,  # Set dynamically from question
        # OFFICIAL: TSQA uses sentence format "Based on the given information, the activity/answer is X."
        "output_format": "tsqa_sentence",
        # OFFICIAL: TSQA classification.csv - questions are self-contained
        "task_instruction": """Classify the time series based on the provided context and data.

###INPUT_CONTEXT###

Answer format: Based on the given information, the activity/answer is [your answer].""",
    },
    "tsqa_anomaly": {
        "task_type": "anomaly",
        "benchmark": "TSQA",
        "domain": None,
        "labels": ["Normal Point", "Anomaly Point"],
        # OFFICIAL: TSQA uses sentence format
        "output_format": "tsqa_sentence",
        # OFFICIAL: TSQA anomaly_detection.csv - questions are self-contained
        "task_instruction": """Determine whether there are anomalies in this time series based on the provided context.

###INPUT_CONTEXT###

Options: ['Normal Point', 'Anomaly Point']
Answer format: Based on the given information, this time series includes [your answer].""",
    },
    "tsqa_forecasting": {
        "task_type": "regression",
        "benchmark": "TSQA",
        "domain": None,
        "labels": None,
        "output_format": "tsqa_sentence",
        # OFFICIAL: TSQA forecasting_imputation CSV - questions include context
        "task_instruction": """Forecast the future values of this time series based on the provided context and historical data.

###INPUT_CONTEXT###

Answer format: Based on the given information, the predictions are [value1, value2, ..., valueN].""",
    },
    "tsqa_imputation": {
        "task_type": "regression",
        "benchmark": "TSQA",
        "domain": None,
        "labels": None,
        "output_format": "tsqa_sentence",
        # OFFICIAL: TSQA forecasting_imputation CSV - questions include context
        "task_instruction": """Fill in the missing values (marked as 'X' or 'NaN') in this time series based on the provided context and historical data.

###INPUT_CONTEXT###

Please give FULL time series with missing value imputed.
Answer format: Based on the given information, the full time series with missing value imputed are [value1, value2, ..., valueN].""",
    },
    "tsqa_qa": {
        "task_type": "qa",
        "benchmark": "TSQA",
        "domain": None,
        "labels": None,
        "output_format": "free_text",
        # OFFICIAL: TSQA open_ended_QA.csv - questions are self-contained
        # Format-aware: multiple_choice, true/false, open_ended_question
        "task_instruction": """Answer the question based on the time series data and provided context.

###INPUT_CONTEXT###

{qa_format_instruction}""",
    }
}
