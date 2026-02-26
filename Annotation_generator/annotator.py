from ollama import chat
import chess
import chess.engine
import time
import pandas as pd

def _get_top_lines(fen, num_lines=5, analysis_time=2.0):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci("/path_to_stockfish/stockfish/stockfish-ubuntu-x86-64-avx2") as engine:
        engine.configure({"Threads": 12, "Hash": 2048})
        analysis = engine.analyse(
            board,
            chess.engine.Limit(time=analysis_time, depth=30),  # Single combined limit
            multipv=num_lines  # Keyword argument
        )
        
        lines = []
        for info in analysis:
            if "pv" not in info:
                continue
                
            moves = [move.uci() for move in info["pv"]]
            lines.append({
                "score": info["score"].relative,
                "moves": moves
            })
            
        return lines[:num_lines]

def _generate_ascii_board(fen):
    pieces = fen.split()[0]
    rows = []
    for row in pieces.split('/'):
        expanded = ''.join(int(c) * '.' if c.isdigit() else c for c in row)
        rows.append(' | '.join(expanded))
    sep = '+---' * 8 + '+'
    return '\n'.join(sep + '\n| ' + r + ' |' for r in rows) + '\n' + sep


def _Stockfish_prompt(fen_position):
    start = time.time()
    top_lines = _get_top_lines(fen_position, num_lines=3)
    end = time.time()
    # print(f"execution time is {end-start}")

    fen_string = fen_position
    side_to_move = 'white' if fen_position.split()[1] == 'w' else 'black'
    best_move = top_lines[0]["moves"][0] if top_lines else None

    # print(side_to_move)
    # print(best_move)

    line_1 = " ".join(top_lines[0]["moves"])
    line_2 = " ".join(top_lines[1]["moves"])
    line_3 = " ".join(top_lines[2]["moves"])

    # print(line_1)

    def format_score(score):
        if score.is_mate():
            return f"Mate({score.mate()})"
        return f"Cp({score.score()})"

    cp_1 = format_score(top_lines[0]["score"]) if len(top_lines) > 0 else None
    cp_2 = format_score(top_lines[1]["score"]) if len(top_lines) > 1 else None
    cp_3 = format_score(top_lines[2]["score"]) if len(top_lines) > 2 else None

    ascii_board = _generate_ascii_board(fen=fen_string)

    prompt = _crafted_prompt(fen_string, best_move, line_1, line_2, line_3, cp_1=cp_1, cp_2=cp_2, cp_3=cp_3, ascii_board=ascii_board)
    return prompt



def _crafted_prompt(FEN, best_move, line_1, line_2, line_3, cp_1, cp_2, cp_3, ascii_board):
    user_prompt = f"""Given a board's FEN string: 
{FEN}

The ASCII board for the given FEN string is:
{ascii_board}

Use the below Centipawn loss (Cp) and move sequence to guide your reasoning:

Line 1; {cp_1}: {line_1}
Line 2; {cp_2}: {line_2}
Line 3; {cp_3}: {line_3}

The best move is : {best_move}

Give reasoning explaining why this is the best move basing your answer on the given information"""

    return user_prompt

def ollama_response(fen):

    user_prompt = _Stockfish_prompt(fen_position=fen)
    
    sys_prompt = """You are a professional chess reasoning agent.
    Answer only as a paragraph without any bullet points or any emotes. 
    Never mention centipawn scores or any other information provided for guiding you, only use them to guide you into a good reasoning.

    When reasoning, use the ASCII board provided by the user to reason why a move is good.

    Be concise and informative, explaining why a best move is played, analyzing both tactically and positionally. 
    Answer with "Reasoning" : <reason> 
    """
    
    response = chat(    
        model='qwen3.5:cloud',
        messages=[
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={
            'temperature':0.6,
            'top_p':0.95, 
            'top_k':20, 
            'min_p':0.0, 
            'presence_penalty':0.0, 
            'repetition_penalty':1.0
        }
    )

    llm_reasoning = response.message.content

    print(response.message.content)

    return user_prompt, llm_reasoning


def main():

    # usr_prompt = Stockfish_prompt(FEN, annotation=annotation)
    # print(usr_prompt)
    # ollama_response(user_prompt=usr_prompt)


    df = pd.read_csv("FEN_Best_moves_100k.csv") 
    
    df_sample = df.sample(n=5, random_state=42)

    df_sample[["prompt", "reasoning"]] = df_sample.apply(
        lambda row: pd.Series(ollama_response(row["FEN"])),
        axis=1
    )

    # Save only required columns (optional)
    output_df = df_sample[["FEN", "prompt", "reasoning"]]

    output_df.to_csv("GRPO_variant/Annotation_generator/lichess_reasonings.csv", index=False)



if __name__ == "__main__":
    main()
