"""
Prompts by Jeongsik and Jae Won
"""

# -------------------------------------------------------
# Instruction
instruction_task = "Your task is to infer the message that the author is trying to convey through the meme. "  # prompt for Jun2024ARR
instruction_format = (
    "The message must be in one single short sentence."  # prompt for Jun2024ARR
)
instruction_format_genBK = (
    "Each background knowledge must be in one single short sentence."
)
instruction_format_genSM = (
    "Surface message must be in one single short sentence."
)

# SM
llm_instruction_SM_noBK = (
    "You will be provided with a meme via a description of a meme (including both its image and text). "
    + instruction_task
    + instruction_format
)
llm_instruction_SM_BK = (
    "You will be provided with a meme via a description of a meme (including both its image and text), and the background knowledge that a reader of the meme needs to possess before they can understand the intent. "
    + instruction_task
    + instruction_format
)
mllm_instruction_SM_noBK = (
    "You will be provided with a meme, and a description of a meme (including both its image and text). "
    + instruction_task
    + instruction_format
)
mllm_instruction_SM_BK = (
    "You will be provided with a meme, a description of a meme (including both its image and text), and the background knowledge that a reader of the meme needs to possess before they can understand the intent. "
    + instruction_task
    + instruction_format
)
# ICtext
llm_instruction_ICtext_noBK = (
    "You will be provided with a meme via a description of its image, and the text written on it. "
    + instruction_task
    + instruction_format
)
llm_instruction_ICtext_BK = (
    "You will be provided with a meme via a description of its image, the text written on it, and the background knowledge that a reader of the meme needs to possess before they can understand the message. "
    + instruction_task
    + instruction_format
)
mllm_instruction_ICtext_noBK = (
    "You will be provided with a meme, the description of its image, and the text written on the meme. "
    + instruction_task
    + instruction_format
)
mllm_instruction_ICtext_BK = (
    "You will be provided with a meme, the description of its image, the text written on the meme, and the background knowledge that a reader of the meme needs to possess before they can understand the message. "
    + instruction_task
    + instruction_format
)

# JWC
# instruction_task_BK = "Your task is to infer the message that background knowledge for the meme"
instruction_task_BK = (
    "You will be provided with a meme. "
    "Your task is to infer the background knowledge that a reader of the meme needs to possess before they can understand "
    "the ultimate intent behind the creation or sharing of a meme, as perceived by its audience. "
    "Background knowledge is the minimum amount of knowledge that is missing from the meme. "
    "It is the knowledge that needs to be combined with visual and textual cues from the meme in order to understand its meaning. "
    "Give me background knowledge in the form of a list. "
    "For example: '1. Soccer is the sports that children likes a lot. 2. There are two main political parties in the US: Democratic and Republican.' "
)
mllm_instruction_gen_BK = (
    ""
    + instruction_task_BK
    + instruction_format_genBK # "The message must be in one single short sentence."
)

instruction_task_SM = (
    "You will be provided with a meme. "
    "Your task is to identify the explicit or surface-level message conveyed by the meme. "
    "The surface message is what the meme is saying directly, including any text, images, or symbols present. "
    "Describe this surface-level message as simply and clearly as possible without interpretation of deeper meaning. "
    # "For example: 'A picture of a dog with sunglasses, captioned “Coolest dog in town,” is directly saying the dog is cool.'" 
)
mllm_instruction_gen_SM = (
    ""
    + instruction_task_SM
    + instruction_format_genSM # "The message must be in one single short sentence."
)
# JWC

# -------------------------------------------------------
# demonstration & input
input_SM = "### Description of a meme (including both its image and text): "
input_IC = "### Description of the image: "
input_image = "### Meme: <image>"
input_text = "### Text on the meme: "
input_BK = "### Background knowledge: "

question_message = "### Message: "
question_bk = "### Background knowledge: "
question_sm = "### Surface Message: "


def newSM(tmp_row):
    if tmp_row["image_caption"][-1] != ".":
        tmp_row["image_caption"] = tmp_row["image_caption"] + "."

    if tmp_row["surface_message"] == "1":
        return (
            tmp_row["image_caption"]
            + " The author describes the image as "
            + "'"
            + tmp_row["text"]
            + "'"
        )
    elif tmp_row["surface_message"] == "2":
        return (
            tmp_row["image_caption"]
            + " The character in the meme says that '"
            + tmp_row["text"]
            + "'"
        )
    else:
        return tmp_row["surface_message"]


class makePrompt_SM:
    @staticmethod
    def SM_llama_zero(SM):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_SM + f"{SM}\n" + question_message,
            },
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def SM_llama_few(SM, messages):  # SM, messages should be a list
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_noBK})
        for i in range(len(messages)):
            prompt.append(
                {"role": "user", "content": input_SM + f"{SM[i]}\n" + question_message}
            )
            prompt.append({"role": "assistant", "content": f"{messages[i]}"})
        prompt.append(
            {
                "role": "user",
                "content": input_SM + f"{SM[len(messages)]}\n" + question_message,
            }
        )
        # prompt.append({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def SM_llama_finetune(example):
        SM = newSM(example)
        message = example["implicit_message"]

        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_SM + SM + "\n" + question_message,
            },
        )
        prompt.append(
            {
                "role": "assistant",
                "content": message,
            }
        )
        return prompt

    @staticmethod
    def SM_llava_zero(SM):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_SM_noBK
                        + "\n\n"
                        + input_SM
                        + f"{SM}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt

    @staticmethod
    def SM_llava_few(SM, messages):  # SM, messages should be a list
        prompt = []
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": mllm_instruction_SM_noBK
                            + "\n\n"
                            + input_SM
                            + f"{SM[i]}\n"
                            + question_message,
                        },
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[i]},
                    ],
                },
            )
        prompt.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_SM_noBK
                        + "\n\n"
                        + input_SM
                        + f"{SM[len(messages)]}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        )
        return prompt

    @staticmethod
    def SM_llava_finetune(example):
        SM = newSM(example)
        message = example["implicit_message"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_SM_noBK
                        + "\n\n"
                        + input_SM
                        + f"{SM}"
                        + "\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{message}"},
                ],
            },
        ]
        return prompt


class makePrompt_SMBK:
    @staticmethod
    def SMBK_llama_zero(SM, BK):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_BK})
        prompt.append(
            {
                "role": "user",
                "content": input_SM
                + f"{SM}"
                + "\n"
                + input_BK
                + f"{BK}"
                + "\n"
                + question_message,
            },
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def SMBK_llama_few(SM, BK, messages):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_BK})
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": input_SM
                    + f"{SM[i]}\n"
                    + input_BK
                    + f"{BK[i]}\n"
                    + question_message,
                }
            )
            prompt.append({"role": "assistant", "content": f"{messages[i]}"})
        prompt.append(
            {
                "role": "user",
                "content": input_SM
                + f"{SM[len(messages)]}\n"
                + input_BK
                + f"{BK[len(messages)]}\n"
                + question_message,
            }
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "conntent": "",
        #     }
        # )
        return prompt

    @staticmethod
    def SMBK_llama_finetune(example):
        SM = newSM(example)
        BK = example["background_knowledge"]
        message = example["implicit_message"]
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_SM_BK})
        prompt.append(
            {
                "role": "user",
                "content": input_SM
                + SM
                + "\n"
                + input_BK
                + BK
                + "\n"
                + question_message,
            },
        )
        prompt.append(
            {
                "role": "assistant",
                "content": message,
            }
        )
        return prompt

    @staticmethod
    def SMBK_llava_zero(SM, BK):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_SM_BK
                        + "\n\n"
                        + input_SM
                        + f"{SM}\n"
                        + input_BK
                        + f"{BK}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt

    @staticmethod
    def SMBK_llava_few(SM, BK, messages):
        prompt = []
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": mllm_instruction_SM_BK
                            + "\n\n"
                            + input_SM
                            + f"{SM[i]}\n"
                            + input_BK
                            + f"{BK[i]}\n"
                            + question_message,
                        },
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[i]},
                    ],
                },
            )
        prompt.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": llm_instruction_SM_BK
                        + "\n\n"
                        + input_SM
                        + f"{SM[len(messages)]}\n"
                        + input_BK
                        + f"{BK[len(messages)]}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        )

        return prompt

    @staticmethod
    def SMBK_llava_finetune(example):
        SM = newSM(example)
        BK = example["background_knowledge"]
        messasge = example["implicit_message"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_SM_BK
                        + "\n\n"
                        + input_SM
                        + f"{SM}"
                        + "\n"
                        + input_BK
                        + f"{BK}"
                        + "\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{message}"},
                ],
            },
        ]
        return prompt


class makePrompt_ICtext:
    @staticmethod
    def ICtext_llama_zero(IC, text):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC}\n"
                + input_text
                + f"{text}\n"
                + question_message,
            },
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def ICtext_llama_few(IC, text, messages):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": input_IC
                    + f"{IC[i]}\n"
                    + input_text
                    + f"{text[i]}\n"
                    + question_message,
                }
            )
            prompt.append({"role": "assistant", "content": f"{messages[i]}"})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC[len(messages)]}\n"
                + input_text
                + f"{text[len(messages)]}\n"
                + question_message,
            }
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def ICtext_llama_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        message = example["implicit_message"]
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + IC
                + "\n"
                + input_text
                + text
                + "\n"
                + question_message,
            },
        )
        prompt.append(
            {
                "role": "assistant",
                "content": message,
            }
        )
        return prompt

    @staticmethod
    def ICtext_llava_zero(IC, text):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_noBK
                        + "\n\n"
                        + input_IC
                        + f"{IC}\n"
                        + input_text
                        + f"{text}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt

    @staticmethod
    def ICtext_llava_few(IC, text, messages):
        prompt = []
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": mllm_instruction_ICtext_noBK
                            + "\n\n"
                            + input_IC
                            + f"{IC[i]}\n"
                            + input_text
                            + f"{text[i]}\n"
                            + question_message,
                        },
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[i]},
                    ],
                },
            )
        prompt.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_noBK
                        + "\n\n"
                        + input_IC
                        + f"{IC[len(messages)]}\n"
                        + input_text
                        + f"{text[len(messages)]}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        )

        return prompt

    @staticmethod
    def ICtext_llava_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        message = example["implicit_message"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_noBK
                        + "\n\n"
                        + input_IC
                        + f"{IC}"
                        + "\n"
                        + input_text
                        + f"{text}"
                        + "\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{message}"},
                ],
            },
        ]
        return prompt


class makePrompt_ICtextBK:
    @staticmethod
    def ICtextBK_llama_zero(IC, text, BK):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_BK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC}"
                + "\n"
                + input_text
                + f"{text}"
                + "\n"
                + input_BK
                + f"{BK}"
                + "\n"
                + question_message,
            },
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def ICtextBK_llama_few(IC, text, BK, messages):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_BK})
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": input_IC
                    + f"{IC[i]}\n"
                    + input_text
                    + f"{text[i]}\n"
                    + input_BK
                    + f"{BK[i]}\n"
                    + question_message,
                }
            )
            prompt.append({"role": "assistant", "content": f"{messages[i]}"})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC[len(messages)]}\n"
                + input_text
                + f"{text[len(messages)]}\n"
                + input_BK
                + f"{BK[len(messages)]}\n"
                + question_message,
            }
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod
    def ICtextBK_llama_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        BK = example["background_knowledge"]
        message = example["implicit_message"]
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_BK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + IC
                + "\n"
                + input_text
                + text
                + "\n"
                + input_BK
                + BK
                + "\n"
                + question_message,
            },
        )
        prompt.append(
            {
                "role": "assistant",
                "content": message,
            }
        )
        return prompt

    @staticmethod
    def ICtextBK_llava_zero(IC, text, BK):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_BK
                        + "\n\n"
                        + input_IC
                        + f"{IC}\n"
                        + input_text
                        + f"{text}\n"
                        + input_BK
                        + f"{BK}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt

    @staticmethod
    def ICtextBK_llava_few(IC, text, BK, messages):
        prompt = []
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": mllm_instruction_ICtext_BK
                            + "\n\n"
                            + input_IC
                            + f"{IC[i]}\n"
                            + input_text
                            + f"{text[i]}\n"
                            + input_BK
                            + f"{BK[i]}\n"
                            + question_message,
                        },
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[i]},
                    ],
                },
            )
        prompt.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_BK
                        + "\n\n"
                        + input_IC
                        + f"{IC[len(messages)]}\n"
                        + input_text
                        + f"{text[len(messages)]}\n"
                        + input_BK
                        + f"{BK[len(messages)]}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        )

        return prompt

    @staticmethod
    def ICtextBK_llava_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        BK = example["background_knowledge"]
        message = example["implicit_message"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_BK
                        + "\n\n"
                        + input_IC
                        + f"{IC}"
                        + "\n"
                        + input_text
                        + f"{text}"
                        + "\n"
                        + input_BK
                        + f"{BK}"
                        + "\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{message}"},
                ],
            },
        ]
        return prompt


# JWC
class makePrompt_genBK:
    @staticmethod # need to change
    def genBK_llama_zero(IC, text):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC}\n"
                + input_text
                + f"{text}\n"
                + question_message,
            },
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod # need to change
    def genBK_llama_few(IC, text, messages):
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": input_IC
                    + f"{IC[i]}\n"
                    + input_text
                    + f"{text[i]}\n"
                    + question_message,
                }
            )
            prompt.append({"role": "assistant", "content": f"{messages[i]}"})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + f"{IC[len(messages)]}\n"
                + input_text
                + f"{text[len(messages)]}\n"
                + question_message,
            }
        )
        # prompt.append(
        #     {
        #         "role": "assistant",
        #         "content": "",
        #     }
        # )
        return prompt

    @staticmethod # need to change
    def genBK_llama_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        message = example["implicit_message"]
        prompt = []
        prompt.append({"role": "system", "content": llm_instruction_ICtext_noBK})
        prompt.append(
            {
                "role": "user",
                "content": input_IC
                + IC
                + "\n"
                + input_text
                + text
                + "\n"
                + question_message,
            },
        )
        prompt.append(
            {
                "role": "assistant",
                "content": message,
            }
        )
        return prompt

    @staticmethod
    def genBK_llava_zero():
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_gen_BK
                        + "\n\n"
                        + question_bk, # "### Message: "
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt

    @staticmethod # need to change
    def genBK_llava_few(IC, text, messages):
        prompt = []
        for i in range(len(messages)):
            prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": mllm_instruction_ICtext_noBK
                            + "\n\n"
                            + input_IC
                            + f"{IC[i]}\n"
                            + input_text
                            + f"{text[i]}\n"
                            + question_message,
                        },
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[i]},
                    ],
                },
            )
        prompt.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_noBK
                        + "\n\n"
                        + input_IC
                        + f"{IC[len(messages)]}\n"
                        + input_text
                        + f"{text[len(messages)]}\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        )

        return prompt

    @staticmethod # need to change
    def genBK_llava_finetune(example):
        IC = example["image_caption"]
        text = example["text"]
        message = example["implicit_message"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_ICtext_noBK
                        + "\n\n"
                        + input_IC
                        + f"{IC}"
                        + "\n"
                        + input_text
                        + f"{text}"
                        + "\n"
                        + question_message,
                    },
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{message}"},
                ],
            },
        ]
        return prompt
# JWC
class makePrompt_genSM:
    @staticmethod
    def genSM_llava_zero():
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": mllm_instruction_gen_SM
                        + "\n\n"
                        + question_sm, # "### Message: "
                    },
                    {"type": "image"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": ""},
            #     ],
            # },
        ]
        return prompt
        