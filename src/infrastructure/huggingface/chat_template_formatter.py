from domain.interfaces.prompt_formatter import PromptFormatter


class ChatTemplateFormatter(PromptFormatter):

    def __init__(self, tokenizer):

        self._tokenizer = tokenizer


    def format_batch(self, batch):

        texts = []

        for question, answer in zip(
            batch["question"],
            batch["answer"]
        ):

            messages = [

                {
                    "role": "user",
                    "content": question
                },

                {
                    "role": "assistant",
                    "content": answer
                }

            ]

            formatted = self._tokenizer.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=False

            )

            texts.append(formatted)

        return {

            "texts": texts

        }