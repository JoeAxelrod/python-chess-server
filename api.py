from flask import Flask, request, jsonify
import chess
import sqlite3

from main import ChessNet, load_model, search, board_to_tensor, board_screenshot_base64, board_matrix

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("chess_games.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY,
        phone TEXT NOT NULL,
        board TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


init_db()


@app.route('/api/start_game', methods=['POST'])
def start_game():
    # phone = request.form.get('phone')
    req_data = request.get_json()
    phone = req_data.get('phone', "")
    if not phone:
        return jsonify({"error": "Phone number is required"}), 400

    board = chess.Board()
    conn = sqlite3.connect("chess_games.db")
    cursor = conn.cursor()

    cursor.execute("INSERT INTO games (phone, board) VALUES (?, ?)", (phone, board.fen()))
    conn.commit()
    conn.close()

    legal_moves = [move.uci() for move in board.legal_moves]

    return jsonify({
        "message": "Game started",
        "board": board.fen(),
        "board_img_base64": board_screenshot_base64(board),
        "legal_moves": legal_moves,
        "board_matrix": board_matrix(board),
    }), 201


@app.route('/api/make_move', methods=['POST'])
def make_move():
    req_data = request.get_json()
    phone = req_data.get('phone', "")
    move = req_data.get('move', "")
    # phone = request.form.get('phone')
    # move = request.form.get('move')

    if not phone or not move:
        return jsonify({
            "error": "Phone number and move are required"
        }), 400

    conn = sqlite3.connect("chess_games.db")
    cursor = conn.cursor()

    cursor.execute("SELECT board FROM games WHERE phone = ?", (phone,))
    result = cursor.fetchone()

    if not result:
        return jsonify({"error": "Game not found"}), 404

    board_fen = result[0]
    board = chess.Board(board_fen)

    try:
        board.push_san(move)
    except ValueError:
        return jsonify({"error": "Invalid move"}), 400

    # Make AI move here
    net = load_model()
    ai_move = search(board, net, depth=3)
    board.push(ai_move)

    cursor.execute("UPDATE games SET board = ? WHERE phone = ?", (board.fen(), phone))
    conn.commit()
    conn.close()

    legal_moves = [move.uci() for move in board.legal_moves]

    return jsonify({
        "message": "Move made",
        "board": board.fen(),
        "board_img_base64": board_screenshot_base64(board),
        "legal_moves": legal_moves,
        "board_matrix": board_matrix(board),
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008)
    # app.run(debug=True)
