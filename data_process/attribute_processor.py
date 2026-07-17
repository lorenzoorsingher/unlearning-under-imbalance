import random

import numpy as np
import pandas as pd

class AttributeProcessor:
    """
    This class contains methods to process attributes and generate training and evaluation data.

    It includes methods to format training questions and answers, as well as methods to format evaluation tasks such as fill-the-blank, classification, and generation. In order to include additional categories some steps are required:
    1. Add the new category to the self.categories list in the __init__ method.
    2. Implement the set_<category> method to format the training question and answer for the new category.
    3. Implement the format_<category> method to format the evaluation task for the new category.
    4. Provide the category templates in the template_questions.json and template_masks.json files, ensuring they are properly formatted and include all necessary placeholders.
    5. Don't forget to add the new category to SENSIBLE_ATTR if it is a sensible attribute, so that it is not included in the training set or
    SKIP_ATTR if it is an attribute that should not be included in the dataset at all (e.g. "medicalconditions", "knownfor" for non-celebrities).

    """

    def __init__(self, sensible_groups):
        """
        Initialize the AttributeProcessor with categories and their corresponding functions.
        the self.categories is a list of tuples, where each tuple contains:
            - category name (str): the name of the category, this is used as key in the dataset and templates
            - set function (callable):  a function that takes question/answer templates and the GT values and returns the modified Q/A
                                        these functions format the data for the training QA, it edits the question and answer strings
                                        and applies the random transformations if specified, the returned value is a tuple (question, answer)
            - get function (callable):  a function that take the standardized GT values and returns a list of differently formatted values
                                        that the model might output as answer to the mask task. Since we don't know a priori how the model
                                        will answer (feet/metric, DD/MM/YYYY, etc.), we provide a list of possible answers
            - description (str): a description of the category, this can be used when prompting to hint the model about the category, some are
                                 slightly different from the category name, e.g. "date of birth" instead of "dateofbirth"
        """
        self.categories = [
            ("born", self.set_location, None, "birthplace"),
            ("residence", self.set_location, None, "residence"),
            ("educatedat", self.set_education, None, "education"),
            ("name", self.set_name, None, "name"),
            ("annualsalary", self.set_salary, None, "salary"),
            ("height", self.set_height, None, "height"),
            ("dateofbirth", self.set_date_of_birth, None, "date of birth"),
            ("employment", self.set_employment, None, "employment"),
            ("politics", self.set_politics, None, "political leaning"),
            ("relationship", self.set_relationship, None, "relationship status"),
        ]

        self.implicit_attributes = [
            ("agegroup", self.set_age_group, None, "age group"),
            ("race", self.set_race, None, "race"),
            ("gender", self.set_gender, None, "gender"),
        ]

        if sensible_groups:
            self.categories += self.implicit_attributes

        # reverse lookup for the categories list
        self.categories_lookup = {k: (v1, v2, v3) for k, v1, v2, v3 in self.categories}

    def set_attribute(self, attribute, q, a, values):
        return self.categories_lookup[attribute][0](q, a, values)

    def get_attr_description(self, attribute):
        return self.categories_lookup[attribute][2]

    def capitalize_all(self, text):
        """
        Capitalize the first letter of each word in the text.
        """
        return " ".join(word.capitalize() for word in text.split())

    def capitalize(self, text):
        """
        Capitalize the first letter of the text.
        """
        return text[0].upper() + text[1:]

    def adjust_pronouns(self, sentence, gender):

        if "<GENDER" in sentence:
            tag = sentence.split("<GENDER")[1].split(">")[0].strip()
            genders = tag.split("/")
            if gender == "male":
                pronoun = genders[0]
            else:
                pronoun = genders[1]

            sentence = sentence.replace(f"<GENDER {tag}>", pronoun)
        return sentence

    # FUNCTIONS TO FORMAT TRAINING QA
    def set_location(self, question, answer, values):
        city, ctry, _ = values
        cc = f"{self.capitalize_all(city)}, {self.capitalize_all(ctry)}"
        c_p = f"{self.capitalize_all(city)}({self.capitalize_all(ctry)})"
        c_ = f"{self.capitalize_all(city)}"
        gts = [cc, c_, c_p]

        # ------------------

        gt = np.random.choice(gts, p=[0.5, 0.35, 0.15])
        answer = answer.replace("<ANSWER>", gt)

        return question, answer, gts

    def set_education(self, question, answer, values):
        education = values
        gt = f"the {self.capitalize_all(education)}"
        answer = answer.replace("<ANSWER>", gt)
        return question, answer, [gt]

    def set_name(self, question, answer, values):
        name = values
        gt = f"{self.capitalize_all(name)}"
        answer = answer.replace("<ANSWER>", gt)
        return question, answer, [gt]

    def set_gender(self, question, answer, values):
        gender = values
        if gender == "female":
            gts = ["female", "woman", "girl"]
        elif gender == "male":
            gts = ["male", "man", "boy"]
        answer = answer.replace("<ANSWER>", gender)
        return question, answer, gts

    def set_salary(self, question, answer, values):
        bin = values
        if bin is not None:
            gt = bin
            answer = answer.replace("<ANSWER>", gt)
        else:
            gt = random.choice(
                [
                    "The salary is not disclosed",
                    "The salary is not available",
                    "The salary is not provided",
                ]
            )
            answer = gt
        return question, answer, [gt]

    def set_height(self, question, answer, values):
        height = values
        answer = answer.replace("<ANSWER>", height)
        return question, answer, [height]

    def set_date_of_birth(self, question, answer, values):
        parsed_date = pd.to_datetime(values)
        formats = ["%Y-%m-%d", "%B %d %Y"]  # , "%Y.%m.%d", "%Y%m%d"]
        gts = [f"{parsed_date.strftime(f)}" for f in formats]
        gt = random.choice(gts)
        answer = answer.replace("<ANSWER>", gt)
        return question, answer, gts

    def set_race(self, question, answer, values):
        answer = answer.replace("<ANSWER>", values)
        return question, answer, [values]

    def set_age_group(self, question, answer, values):

        answer = answer.replace("<ANSWER>", values)
        return question, answer, [values]

    def set_employment(self, question, answer, values):
        employment = values
        # gt = f"{self.capitalize_all(employment)}"
        answer = answer.replace("<ANSWER>", employment)
        return question, answer, [employment]

    def set_politics(self, question, answer, values):
        leaning = values
        answer = answer.replace("<ANSWER>", leaning)
        return question, answer, [leaning]

    def set_relationship(self, question, answer, values):
        status = values
        answer = answer.replace("<ANSWER>", status)
        return question, answer, [status]


