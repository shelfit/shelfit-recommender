class StringFormatterUtil:
    @staticmethod
    def remove_json_markdown(model_response: str)-> str:
        if model_response.startswith('```'):
            model_response = model_response[3:]

            if model_response[:4].lower() == 'json':
                model_response = model_response[4:]

            model_response = model_response[:-3]

        return model_response