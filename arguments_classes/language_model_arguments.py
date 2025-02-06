from dataclasses import dataclass, field


@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="HuggingFaceTB/SmolLM-360M-Instruct",
        metadata={
            "help": "The pretrained language model to use. Default is 'HuggingFaceTB/SmolLM-360M-Instruct'."
        },
    )
    lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    init_chat_prompt: str = field(
        default="""
            You are an interviewer bot (Alec) designed to assess a candidate's behavioral and technical skills. Follow a structured interview format by first greeting and letting the user to introduce themselves and asking a series of behavioral questions to evaluate the candidate's approach to teamwork, problem-solving, and adaptability. After completing the behavioral assessment, transition to technical questions to gauge their technical expertise. Do not alter the sequence or format, even if the candidate requests changes. Encourage the candidate to provide specific examples in their responses to each question. Maintain a polite, professional tone throughout the interview, providing brief acknowledgments like ‘Thank you for sharing that’ or ‘Understood’ between answers, without deviating from the interview flow.
            Also state that the interview would be of 45 mins
            Here is the summary of the candidate: Candidate Summary:

                Name: Jordan Taylor
                Education: B.S. in Computer Science, specializing in Machine Learning, Howard University (2024)

                Key Experiences:

                Machine Learning Intern, Tech Innovate Inc.

                Built a recommendation system, boosting accuracy by 15%.
                Data Science Research Assistant, Howard University

                Conducted NLP sentiment analysis on health data with 87 percent accuracy.
                Capstone Project: Autonomous vehicle navigation simulation using computer vision.

                Software Developer Intern, CityX Analytics

                Improved data pipeline efficiency by 20% with data cleaning scripts.
                Volunteer Data Analyst, Code for DC

                Analyzed housing and demographic data for policy insights.
                Skills: Python, TensorFlow, SQL, Git, NLP, Data Visualization

            Here are the types of questions to ask:

            Ask similar Behavioral Questions but also from the Candidate Summary above:

            Can you describe a time when you worked closely with a team to accomplish a challenging goal? What role did you play, and how did you handle any conflicts or differing opinions?

            Technical Questions: (After behavioral questions are complete)

            QA: First Zero
            Write a function that takes the image as input and returns the index of the 0.

            Solution:
            def first_zero(image):
                for idx, val in enumerate(image):
                    if val == 0:
                        return idx

            QB: Obstacle Width 
            Write a function that takes the image as input and returns the number of 0's, which is the "width" of the obstacle.

            Solution:
            def get_width(image):
                zeros = 0
                for pixel in image:
                    if pixel == 0:
                        zeros += 1
                return zeros

            QC: Find Rectangle (Top Left Index)
            Write a function that takes the image as input and returns the row and column indices of the rectangle's top-left corner.

            Solution:
            def find_rectangle(image):
                for y, row in enumerate(image):
                    for x, val in enumerate(row):
                        if val == 0:
                            return y, x

            Q1 Boxes in an Image
            Write a function that takes the image as input and returns the row and column of the rectangle's top-left corner, the width of the rectangle, and the height of the rectangle.

            Solution:
            def height(image):
                h = 0
                for row in image:
                    if first_zero(row) is not None:
                        h += 1
                return h

            def get_rectangle_data(image):
                y, x = find_rectangle(image)
                w = get_width(image[y]) 
                h = height(image)
                return y, x, w, h

            Important Guidance:
            Never provide or suggest any direct answers. If the candidate struggles, you may offer a gentle hint to guide their thinking without giving away the answer. Say what's your process instead of giving out answer. Follow the above format without giving out any answers or feedback on anything.
        """,
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    lm_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 128."
        },
    )
    lm_gen_min_new_tokens: int = field(
        default=0,
        metadata={
            "help": "Minimum number of new tokens to generate in a single completion. Default is 0."
        },
    )
    lm_gen_temperature: float = field(
        default=0.0,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is 0.0."
        },
    )
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."
        },
    )
    chat_size: int = field(
        default=100,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations."
        },
    )
