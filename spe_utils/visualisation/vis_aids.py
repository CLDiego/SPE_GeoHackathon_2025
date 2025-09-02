from IPython.display import HTML, display
from transformers import BertTokenizer

def bert_tokenize_and_color(text, tokenizer ):
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FFD700', '#00CED1', '#FF00FF', '#FFFF00',
              '#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF1493', '#8A2BE2',
              '#FF8C00', '#228B22', '#DC143C', '#32CD32', '#1E90FF', '#FFD700', '#FF69B4']

    tokens = tokenizer.tokenize(text)
    colored_html = ""
    
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        # Replace special characters for display
        display_token = token.replace('Ġ', '▁')  # GPT-2 uses Ġ for spaces
        colored_html += f'<span style="background-color:{color}; color: white; padding: 2px 4px; margin: 1px; border-radius: 3px;">{display_token}</span>'
    
    print(f"Original text: {text}")
    print(f"Number of tokens: {len(tokens)}")
    display(HTML(colored_html))
    print(f"Tokens: {tokens}")
    print("-" * 80)