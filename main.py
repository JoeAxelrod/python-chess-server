import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
from stockfish import Stockfish
import random
import torch
import chess.svg
# from IPython.display import SVG, display
import pickle
import os
import io
import base64
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish"


def load_prepared_data(file_path='chess_data.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        print(f"File '{file_path}' does not exist.")
        return None


def piece_to_channel(piece):
    piece_dict = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    return piece_dict.get(piece.symbol(), -1)


def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            channel = piece_to_channel(piece)
            row, col = divmod(i, 8)
            tensor[channel, row, col] = 1

    return tensor


# Define the neural network architecture
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * 8 * 8, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_model(model_path='chess_model.pth'):
    net = ChessNet()

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    net.eval()
    return net


# Prepare the data for training
def prepare_data(num_positions=5000, max_depth=10, file_path='chess_data.pkl'):
    prepared_data = load_prepared_data(file_path)
    if prepared_data is not None:
        print("Loaded prepared data.")
        return prepared_data
    print("Preparing data...")
    stockfish = Stockfish(STOCKFISH_PATH)
    positions = []
    evaluations = []

    board = chess.Board()
    print(board.legal_moves)
    print(board.fen())

    for _ in range(num_positions):
        if _ % 100 == 0:
            print(f"Generating position {_} / {num_positions}")
        board = chess.Board()
        depth = random.randint(1, max_depth)

        for _ in range(depth):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            board.push(move)

        # Get the evaluation from Stockfish
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()

        # Normalize the evaluation to be in the range of [-1, 1]
        if evaluation['type'] == "cp":
            normalized_eval = evaluation['value'] / 100.0
        else:  # "mate"
            normalized_eval = 1.0 if evaluation['value'] > 0 else -1.0

        positions.append(board_to_tensor(board))
        evaluations.append(normalized_eval)

    # Convert to tensors
    positions = torch.stack(positions)
    evaluations = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)

    # Split into train and validation sets
    split_idx = int(0.8 * num_positions)
    train_positions = positions[:split_idx]
    train_evaluations = evaluations[:split_idx]
    val_positions = positions[split_idx:]
    val_evaluations = evaluations[split_idx:]

    # Save the prepared data to a file
    with open(file_path, 'wb') as f:
        pickle.dump(((train_positions, train_evaluations), (val_positions, val_evaluations)), f)
        print(f"Prepared data saved to '{file_path}'.")

    return (train_positions, train_evaluations), (val_positions, val_evaluations)


# def save_board_image(board, file_name='board.png'):
#     svg_board = chess.svg.board(board=board)
#     drawing = svg2rlg(io.StringIO(svg_board))
#     renderPM.drawToFile(drawing, file_name, fmt='PNG')


def train(net, data, epochs, batch_size):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_positions, train_evaluations = data

    print("train_positions.size()[0]", train_positions.size()[0])
    for epoch in range(epochs):
        permutation = torch.randperm(train_positions.size()[0])
        for i in range(0, train_positions.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_positions, batch_evaluations = train_positions[indices], train_evaluations[indices]

            outputs = net(batch_positions)
            loss = criterion(outputs, batch_evaluations)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    torch.save(net.state_dict(), 'chess_model.pth')


def minimax(board, net, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        board_tensor = board_to_tensor(board)
        evaluation = net(board_tensor.unsqueeze(0)).item()
        return evaluation

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, net, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, net, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval


def search(board, net, depth):
    best_move = None
    best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
    maximizing_player = board.turn == chess.WHITE

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, net, depth - 1, not maximizing_player)
        board.pop()

        if maximizing_player and eval > best_eval:
            best_eval = eval
            best_move = move
        elif not maximizing_player and eval < best_eval:
            best_eval = eval
            best_move = move

    return best_move


# Integrate the AI into a game loop
def play_game(net):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            print(board)
            print(board.legal_moves)
            move = input("Enter your move: ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move. Try again.")
                continue
        else:
            move = search(board, net, depth=3)
            board.push(move)
        # board_screenshot_base64(board)

        print("\n")
    # display(SVG(chess.svg.board(board=board)))

def board_matrix(board):
    matrix = [['.' for _ in range(8)] for _ in range(8)]

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            matrix[row][col] = piece.symbol()
    # return json.dumps(matrix)
    return matrix


def board_screenshot_base64(board):
    # svg_board = chess.svg.board(board=board, size=300)
    svg_board = chess.svg.board(board=board)
    base64_data = base64.b64encode(svg_board.encode("utf-8"))
    return base64_data.decode("utf-8")


def board_screenshot_base64_png(board):
    svg_board = chess.svg.board(board=board)
    drawing = svg2rlg(io.StringIO(svg_board))
    output = BytesIO()
    renderPM.drawToFile(drawing, output, fmt='PNG')
    base64_data = base64.b64encode(output.getvalue())
    return base64_data.decode("utf-8")


if __name__ == "__main__":
    net = load_model()
    train_data, val_data = prepare_data(num_positions=5000, file_path='chess_data.pkl')
    train(net, train_data, epochs=2, batch_size=32)
    play_game(net)
