# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:27:03 2018

The Game fo Tic Tac Toe

@author: Santhosh B
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout
import random
import pandas as pd
import matplotlib.pyplot as plt

class tic_tac_toe():
    def __init__(self):
        '''Initialize a 3x3 numpy array which is the tic tac toe board, with place holder as 2
        '0' denotes O
        '1' denotes X'''
        self.board = np.full((3,3),2)
    
    def toss(self):
        ''' Function to simulate toss, to pick which player will first play
        
        returns: 
            Returns 1 if player assigned '1' has won the toss
            Returns 0 if player assigned '0' has won the toss
        '''
        turn = np.random.randint(0,2,size=1)
        
        if turn.mean() == 0:
            self.turn_monitor = 0
        else:
            self.turn_monitor = 1
        
        return self.turn_monitor
    
    def move(self,player,coord):
        '''Function to make the move of particular player
        After the move is done, turn_monitor is flipped
        
        Args:
            player - 1 or 0 depending on the player who is making the move
            coord - Co ordinate of numpy array in which the move has to be made
        
        Returns:
            game_status() - Status of the Game Won/Lost/In progress
            Board - Current board after making the move
        '''
        
        if self.turn_monitor != player or self.game_status()!="In Progress" or self.board[coord] != 2:
            raise ValueError("Invalid Move")
        
        self.board[coord] = player
        self.turn_monitor = 1-player
        
        return self.game_status(),self.board

    def game_status(self):
        '''Function to check whether the Game is won/lost or in progress
        
        Returns: game status'''
        
        #Check for wins along the rows
        for i in range(self.board.shape[0]):
            if 2 not in self.board[i,:] and len(set(self.board[i,:])) == 1:
                return "Won"
        
        #Check for wins along the columns
        for i in range(self.board.shape[1]):
            if 2 not in self.board[:,i] and len(set(self.board[:,i])) == 1:
                return "Won"
            
        #Check for wins along the first Diagonal
        if 2 not in np.diag(self.board) and len(set(np.diag(self.board))) == 1:
            return "Won"
            
        #Check for wins along the second Diagonal
        if 2 not in np.diag(np.fliplr(self.board)) and len(set(np.diag(np.fliplr(self.board)))) == 1:
            return "Won"
        
        #check for Draw
        if 2 not in self.board:
            return "Drawn"
        else:
            return "In Progress"

def legal_move_generator(turn_monitor,board):
    ''' Function which gives all the legal moves a player can make
    
    Args:
        turn_monitor - Tells which player has to make the move
        board - current state of the board
        
    Returns:
        legal_moves - All the possible moves that the player can make. 
                      This Dict has keys as position of board numpy array and value as the flattened board after the player has played
    '''
    legal_moves = {}
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i,j] == 2:
                board_copy = board.copy()
                board_copy[i,j] = turn_monitor
                legal_moves[(i,j)] = board_copy.flatten()
    
    return legal_moves

def move_selector(model,board,turn_monitor):
    '''
    This function first gets all the legal moves the player can make
    Then it uses the neural network to predict the score for each of the move
    Then the move with highest score will get selected
    '''
    tracker = {}
    legal_moves = legal_move_generator(turn_monitor,board)
    for legal_move_coord in legal_moves:
        score = model.predict(legal_moves[legal_move_coord].reshape(1,9))
        tracker[legal_move_coord] = score
    
    selected_move = max(tracker,key = tracker.get)
    next_board_state = legal_moves[selected_move]
    best_score = tracker[selected_move]
    
    return selected_move,next_board_state, best_score

def row_winning_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the player wins if the player's ticker is placed in any legal position along the row
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[0]):
            if 2 not in current_board_state_copy[:,i] and len(set(current_board_state_copy[:,i])) == 1:
                return each_move

def column_winning_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the player wins if the player's ticker is placed in any legal position along the column
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[1]):
            if 2 not in current_board_state_copy[:,i] and len(set(current_board_state_copy[:,i])) == 1:
                return each_move

def diag_1_winning_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the player wins if the player's ticker is placed in any legal position along the first Diagonal
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 2 not in np.diag(current_board_state_copy) and len(set(np.diag(current_board_state_copy))) == 1:
            return each_move

def diag_2_winning_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the player wins if the player's ticker is placed in any legal position along the second Diagonal
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 2 not in np.diag(np.fliplr(current_board_state_copy)) and len(set(np.diag(np.fliplr(current_board_state_copy)))) == 1:
            return each_move

def row_block_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether if program's win is blocked in a row by placing a 0
    
    Returns: Coordinate if the blocking position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[0]):
            if 2 not in current_board_state_copy[:,i] and (current_board_state_copy[:,i] == 1).sum()==2:
                if not(2 not in current_board_state[:,i] and (current_board_state[:,i] == 1).sum()==2):
                    return each_move

def column_block_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether if program's win is blocked in a column by placing a 0
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[1]):
            if 2 not in current_board_state_copy[:,i] and (current_board_state_copy[:,i] == 1).sum()==2:
                if not(2 not in current_board_state[:,i] and (current_board_state[i,:] == 1).sum()==2):
                    return each_move

def diag_1_block_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether if program's win is blocked by placing a 0 along the first Diagonal
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 2 not in np.diag(current_board_state_copy) and (np.diag(current_board_state_copy) == 1).sum()==2:
            if not(2 not in np.diag(current_board_state) and (np.diag(current_board_state) == 1).sum()==2):
                return each_move

def diag_2_block_check(current_board_state,legal_moves,turn_monitor):
    '''
   Function to check whether if program's win is blocked by placing a 0 along the second Diagonal
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 2 not in np.diag(np.fliplr(current_board_state_copy)) and (np.diag(np.fliplr(current_board_state_copy)) == 1).sum()==2:
            if not(2 not in np.diag(np.fliplr(current_board_state)) and (np.diag(np.fliplr(current_board_state)) == 1).sum()==2):
                return each_move


def row_second_move_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the row will have 2 0s and no 1s after making the legal move
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[0]):
            if 1 not in current_board_state_copy[:,i] and (current_board_state_copy[:,i] == 0).sum()==2:
                if not(1 not in current_board_state[:,i] and (current_board_state[:,i] == 0).sum()==2):
                    return each_move

def column_second_move_check(current_board_state,legal_moves,turn_monitor):
    '''--
    Function to check whether the column will have 2 0s and no 1s after making the legal move
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        for i in range(current_board_state_copy.shape[1]):
            if 1 not in current_board_state_copy[:,i] and (current_board_state_copy[:,i] == 0).sum()==2:
                if not(1 not in current_board_state[:,i] and (current_board_state[i,:] == 0).sum()==2):
                    return each_move

def diag_1_second_move_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the first Diagonal will have 2 0s and no 1s after making the legal move
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 1 not in np.diag(current_board_state_copy) and (np.diag(current_board_state_copy) == 0).sum()==2:
            if not(1 not in np.diag(current_board_state) and (np.diag(current_board_state) == 0).sum()==2):
                return each_move

def diag_2_second_move_check(current_board_state,legal_moves,turn_monitor):
    '''
    Function to check whether the second Diagonal will have 2 0s and no 1s after making the legal move
    
    Returns: Coordinate if the winning position is found
    '''
    legal_move_coords = list(legal_moves.keys())
    for each_move in legal_move_coords:
        current_board_state_copy = current_board_state.copy()
        current_board_state_copy[each_move] = turn_monitor
        
        if 1 not in np.diag(np.fliplr(current_board_state_copy)) and (np.diag(np.fliplr(current_board_state_copy)) == 0).sum()==2:
            if not(1 not in np.diag(np.fliplr(current_board_state)) and (np.diag(np.fliplr(current_board_state)) == 0).sum()==2):
                return each_move
            
def opponent_move_selector(board_state,turn_monitor,mode):
    '''
    Function which is used to select the opponents move
    If the mode is easy, the move is randomly selected
    If the mode is hard, following conditions are checked
    1) Move which wins for opponent is selected
    2) Move which blocks for player is selected
    3) Move which forms 2 blocks for opponent is selected
    '''
    legal_moves = legal_move_generator(turn_monitor,board_state)
    
    winning_moves = [row_winning_check,column_winning_check,diag_1_winning_check,diag_2_winning_check]
    block_moves = [row_block_check,column_block_check,diag_1_block_check,diag_2_block_check]
    second_moves = [row_second_move_check,column_second_move_check,diag_1_second_move_check,diag_2_second_move_check]
    
    if mode == "Easy":
        selected_move = random.choice(list(legal_moves.keys()))
        return selected_move
    elif mode == "Hard":
        random.shuffle(winning_moves)
        random.shuffle(block_moves)
        random.shuffle(second_moves)
        
        for fn in winning_moves:
            if fn(board_state,legal_moves,turn_monitor):
                return fn(board_state,legal_moves,turn_monitor)
        
        for fn in block_moves:
            if fn(board_state,legal_moves,turn_monitor):
                return fn(board_state,legal_moves,turn_monitor)
        
        for fn in second_moves:
            if fn(board_state,legal_moves,turn_monitor):
                return fn(board_state,legal_moves,turn_monitor)
            
        selected_move = random.choice(list(legal_moves.keys()))
        return selected_move
  
def shift(scores,pos,newVal):
    '''Function to shift the values in a list by the given pos'''
    length = len(scores)    
    expected_length = length+pos
    new_scores = scores[(length-expected_length):]
    new_scores.append(newVal)
    
    return new_scores

def train(model,mode,printStatus=False):
    '''
    This function plays the game, making move for each player and the resulting board and score are saved
    Then the scores are adjusted depending on the Game result
    Then the model is trained for one game using the set of board positions and corresponding scores
    '''
    
    new_board_state_list = []
    scores_list = []
    corrected_scores_list = []
    game = tic_tac_toe()
    game.toss()
    
    while(1):
      if game.game_status() == "In Progress" and game.turn_monitor == 1:
          selected_move,new_board_state,score   = move_selector(model,game.board,game.turn_monitor)
          new_board_state_list.append(new_board_state)
          scores_list.append(score[0][0])
          game_status,board = game.move(game.turn_monitor,selected_move)
          if printStatus:
              print("Player's Move\n")
              print(board)
              print("\n")
      elif game.game_status() == "In Progress" and game.turn_monitor == 0:
         selected_move = opponent_move_selector(game.board,game.turn_monitor,mode)
         game_status,board = game.move(game.turn_monitor,selected_move)
         if printStatus:
             print("Opponent's Move\n")
             print(board)
             print("\n")
      else:
          break
      
    new_board_state_list = tuple(new_board_state_list)
    new_board_state_list = np.vstack(new_board_state_list)
    if game.game_status() == "Won" and 1-game.turn_monitor==1:
        corrected_scores_list = shift(scores_list,-1,1.0)
        result = "Won"
    if game.game_status() == "Won" and 1-game.turn_monitor==0:
        corrected_scores_list = shift(scores_list,-1,-1.0)
        result = "Lost"
    if game.game_status() == "Drawn":
        corrected_scores_list = shift(scores_list,-1,0)
        result = "Drawn"
    
    if printStatus:
        print("Player has "+result+"\n")
    
    x = new_board_state_list
    y = corrected_scores_list
    
    x = x.reshape(-1,9)
    y = np.array(y)
    
    #Updating weights of model for each game
    model.fit(x,y,epochs=1,batch_size=1,verbose=0)
      
    return model,y,result

def create_train_save_model():
    #Neural Network Model which will be used to train the game
    model = Sequential()
    model.add(Dense(18,input_dim=9,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(9,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,kernel_initializer='normal'))
    
    sgd = SGD(lr=0.001,momentum=0.8)
    model.compile(loss='mean_squared_error',optimizer=sgd)
        
    #Training model for 250000 games
    game_counter = 1
    data_for_graph = pd.DataFrame()
    mode = ['Easy','Hard']
    
    while(game_counter<=100000):
        mode_selected = np.random.choice(mode,1,p=[0.5,0.5])
        model,y,result = train(model,mode_selected,False)
        data_for_graph = data_for_graph.append({'game_counter':game_counter,'result':result},ignore_index=True)
        if game_counter%10000 == 0:
            print(result)
        game_counter+=1
    
    #Visualizing the training improvement
    bins = np.arange(1,game_counter/10000)*10000
    data_for_graph['game_counter_bins'] = np.digitize(data_for_graph['game_counter'],bins,right=True)
    
    counts = data_for_graph.groupby(['game_counter_bins','result']).game_counter.count().unstack()
    
    ax=counts.plot(kind='bar', stacked=True,figsize=(17,5))
    ax.set_xlabel("Count of Games in Bins of 10,000s")
    ax.set_ylabel("Counts of Draws/Losses/Wins")
    
    #Save the training model
    model.save('tic_tac_toe.h5') 













       
            
            
            
            
            
            
            
            