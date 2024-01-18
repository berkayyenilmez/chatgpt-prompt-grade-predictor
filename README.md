# Project Title

## Overview of the Repository

This repository contains a series of scripts and code pieces designed to [briefly describe the main function]. Below is a list of the main components and their roles in the project:

- `hw_score_predict.ipynb`:
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
        last_user_text = None  # Keep track of the last user message

        for i, convo in enumerate(conversations):
            user_div = convo.find("div", attrs={"data-message-author-role": "user"})
            assistant_div = convo.find("div", attrs={"data-message-author-role": "assistant"})

            # When a user message is found, save it to last_user_text
            if user_div:
                last_user_text = user_div.text.strip()
            if assistant_div and assistant_div.find("code"):  # This assumes that <code> tags are used for code blocks
                code_block_count += 1

            # When an assistant message follows a user message, pair them
            if assistant_div and last_user_text is not None:
                convo_texts.append({
                    "role": "user",
                    "text": last_user_text,
                    "response": assistant_div.text.strip()  # Pair with the last user message
                })
                last_user_text = None  # Reset last_user_text after pairing
        total_code_response_list.append((file_code, code_block_count))
        
        code2convos[file_code] = convo_texts
total_code_response_df = pd.DataFrame(total_code_response_list, columns=['code', 'code_responses'])

- `script_name_2.py`: Description of this script's functionality.
- [Additional scripts and their descriptions]

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

