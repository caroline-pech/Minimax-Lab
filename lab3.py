# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    num_cols = board.num_cols
    num_rows = board.num_rows
    total_pieces = num_cols * num_rows
    for chain in board.get_all_chains(current_player=None):
    	if len(chain) >= 4:
        	return True
    if board.count_pieces() == total_pieces:
    	return True
    else:
    	return False


def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    copy_board = board.copy()

    new_boards = []
    if is_game_over_connectfour(board):
    	return []
    else:
	    for col in range(board.num_cols):
	    	if not board.is_column_full(col):
	    		new_boards.append(copy_board.add_piece(col))
	    return new_boards

def is_tie(board):
	"""Only call with endgame board."""
	p1_chains = board.get_all_chains(True)
	p2_chains = board.get_all_chains(False)
	p1_max_chain = max(p1_chains, key = len)
	p2_max_chain = max(p2_chains, key = len)
	if len(p1_max_chain) < 4 and len(p2_max_chain) < 4:
		return True
	else:
		return False

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if is_tie(board):
    	return 0
    elif is_current_player_maximizer:
    	return -1000
    elif not is_current_player_maximizer:
    	return 1000
    else:
    	return None 

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    total_pieces = board.num_cols * board.num_rows
    if is_game_over_connectfour(board) and is_current_player_maximizer:
    	return -1000 - (total_pieces - board.count_pieces())
    elif is_game_over_connectfour(board) and not is_current_player_maximizer:
    	return 1000 + (total_pieces - board.count_pieces())
    else:
    	return 0

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    scores = {1: 1, 2: 10, 3: 100}
    score_p1 = 0
    for chain in board.get_all_chains(is_current_player_maximizer):
    	score_p1 += scores[len(chain)]
    score_p2 = 0
    for chain in board.get_all_chains(not is_current_player_maximizer):
    	score_p2 += scores[len(chain)]
    return score_p1 - score_p2

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def get_paths(state):
	paths = []
	def create_paths(state, path):
		npath = path[:]
		npath.append(state)

		if state.is_game_over():
			paths.append(npath)
			return True

		else:
			for child in state.generate_next_states():
				create_paths(child, npath)

	create_paths(state, [])
	return paths

def dfs_maximizing(state):
	paths = get_paths(state)

	static_evals = 0
	scores = []

	for path in paths:
		static_evals += 1
		result = path[-1]
		score = result.get_endgame_score()
		scores.append(score)

	max_score = max(scores)
	path_of_max_score = paths[scores.index(max(scores))]
	return (path_of_max_score, max_score, static_evals)

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))

def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    if state.is_game_over():
    	return ([state], state.get_endgame_score(maximize), 1)

    children = state.generate_next_states()
    results = []
    static_evals = 0

    for child in children:
    	result = minimax_endgame_search(child, not maximize)
    	results.append(result)

    	static_evals += result[2]

    	if maximize:
    		best = max(results, key = lambda x: x[1])
    	else:
    		best = min(results, key = lambda x: x[1])

    	if result == best:
    		total_best_path = result[0]
    		total_best_path.insert(0, state)
    		max_score = result[1]

    return (total_best_path, max_score, static_evals)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))
def base_checks(state, maximize, heuristic_fn, depth_limit):
    if state.is_game_over():
    	return ([state], state.get_endgame_score(maximize), 1)

    if depth_limit == 0:
    	val = heuristic_fn(state.get_snapshot(), maximize)
    	return ([state], val, 1)
    return None	

def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    base_cases = base_checks(state, maximize, heuristic_fn, depth_limit)
    if base_cases:
    	return base_cases

    children = state.generate_next_states()
    results = []
    static_evals = 0

    for child in children:
    	result = minimax_search(child, heuristic_fn, depth_limit - 1, not maximize)
    	results.append(result)

    	static_evals += result[2]

    	if maximize:
    		best = max(results, key = lambda x: x[1])
    	else:
    		best = min(results, key = lambda x: x[1])

    	if result == best:
    		total_best_path = result[0]
    		total_best_path.insert(0, state)
    		max_score = result[1]

    return (total_best_path, max_score, static_evals)

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    base_cases = base_checks(state, maximize, heuristic_fn, depth_limit)
    if base_cases:
    	return base_cases

    children = state.generate_next_states()
    results = []
    best_path = []
    static_evals = 0
    
    for child in children:
        result = minimax_search_alphabeta(child, alpha, beta, heuristic_fn, depth_limit-1, not maximize)
        results.append(result)

        static_evals += result[2]

        if maximize:
            best = max(results, key=lambda x: x[1])
            alpha = max(alpha, best[1])
            if alpha >= beta:
                return (best_path, alpha, static_evals)

        else:
            best = min(results, key=lambda x: x[1])
            beta = min(beta, best[1])
            if alpha >= beta:
                return (best_path, beta, static_evals)

        if result == best:
            best_path = result[0]
            best_path.insert(0, state)
            max_score = result[1]
        
    return (best_path, max_score, static_evals)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

#pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime = AnytimeValue()
    for d in range(1, depth_limit+1):
        val = minimax_search_alphabeta(state, -INF, INF, heuristic_fn, d, maximize)
        anytime.set_value(val)
    return anytime

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Caroline Pech"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = "12"
WHAT_I_FOUND_INTERESTING = "Theory of the games"
WHAT_I_FOUND_BORING = "I feel like we were not prepared to do this lab."
SUGGESTIONS = "Explain more in class"
