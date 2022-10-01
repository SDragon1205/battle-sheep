from copy import deepcopy
from tkinter import E
from tkinter.messagebox import NO
from unittest import result
import STcpClient
import numpy as np
import random
import math
import cv2
import imageio
import os
import glob
from time import process_time

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''

def InitPos(mapStat, playerID):
    
    '''
        Write your code here

    '''
    sheep = np.zeros([12, 12])
    init_state = State(playerID, mapStat, sheep)
    init_node = Node(init_state)
    current_node = init_node
    next_node = mcts(current_node, 4.5)
    return next_node.state.action_list[-1][0]

'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
'''
def GetStep(playerID, mapStat, sheepStat):
    '''
    Write your code here
    
    '''

    init_state = State(playerID, mapStat, sheepStat)
    init_node = Node(init_state)
    current_node = init_node
    next_node = mcts(current_node, 4.5)
    step = next_node.state.action_list[-1]

    return step


### for the env ###



### for the env ###


# def initialMap():
#     initGameStat = np.zeros((12, 12), dtype=np.int32)

#     # create border
#     temp_map = np.ones((14, 14), dtype=np.int32)
#     temp_map[1:13, 1:13] = np.zeros([12, 12])

#     while True:
#         n_free = 0
#         t = [[7, 7]]
#         prob = 0.7
#         rand = random.random()
#         if rand < prob:
#             # as free
#             n_free += 1
#             temp_map[7][7] = -1
#         else:
#             temp_map[7][7] = 1

#         while n_free < 64:
#             if len(t) == 0 & n_free != 64:
#                 # recreate
#                 print("recreate")
#                 n_free = 0
#                 temp_map[1:13, 1:13] = np.zeros([12, 12])
#                 t = [[7, 7]]
#                 prob = 0.7
#                 rand = random.random()
#                 if rand < prob:
#                     # as free
#                     n_free += 1
#                     temp_map[7][7] = -1
#                 else:
#                     temp_map[7][7] = 1
#                 continue
#             random.shuffle(t)
#             x, y = t.pop()
#             window = temp_map[x - 1:x + 2, y - 1:y + 2]

#             neighbor = []
#             # 3
#             if window[0][1] == 0:
#                 neighbor.append([x - 1, y])
#             # 4
#             if window[2][1] == 0:
#                 neighbor.append([x + 1, y])

#             if y % 2 == 1:
#                 # 1
#                 if window[0][0] == 0:
#                     neighbor.append([x - 1, y - 1])

#                 # 2
#                 if window[1][0] == 0:
#                     neighbor.append([x, y - 1])

#                 # 5
#                 if window[0][2] == 0:
#                     neighbor.append([x - 1, y + 1])

#                 # 6
#                 if window[1][2] == 0:
#                     neighbor.append([x, y + 1])

#             elif y % 2 == 0:
#                 # 1
#                 if window[1][0] == 0:
#                     neighbor.append([x, y - 1])
#                 # 2
#                 if window[2][0] == 0:
#                     neighbor.append([x + 1, y - 1])

#                 # 5
#                 if window[1][2] == 0:
#                     neighbor.append([x, y + 1])
#                 # 6
#                 if window[2][2] == 0:
#                     neighbor.append([x + 1, y + 1])

#             rand = np.random.random(len(neighbor))
#             rand = rand < prob

#             for i in range(len(neighbor)):
#                 m, n = neighbor[i]
#                 if rand[i]:
#                     # as free
#                     n_free += 1
#                     t.append([m, n])
#                     temp_map[m][n] = 1
#                 else:
#                     temp_map[m][n] = -1
#                 if n_free == 64: break

#         n_component, _, _ = getConnectRegion(1, temp_map[1:13, 1:13])
#         if n_component != 1:
#             # print('recreate because not 1-component')
#             temp_map[1:13, 1:13] = np.zeros([12, 12])
#         else:
#             break

#     # fill all hole
#     temp_map[temp_map == 0] = -1

#     initMapStat = temp_map[1:13, 1:13]
#     initMapStat[initMapStat == 1] = 0

#     return initMapStat, initGameStat



def get_reward(map, org_score=False):
    score = [0, 0, 0, 0]
    max_connect = [0, 0, 0, 0]
    visited = np.zeros((12, 12))

    # for x in range(12):
    #     for y in range(12):
    for pos in np.argwhere(map >= 1):
        x, y = pos
        # if map[x, y] >= 1:
        id = int(map[x, y])
        score[id - 1] += 1

    for x in range(12):
        for y in range(12):

            if visited[x, y] == 0 and map[x, y] >= 1:
                i = int(map[x, y])
                tmp_score = 1
                unvisited = [(x, y)]
                visited[x, y] = 1
                while unvisited:
                    # print('bfs')
                    prev_x, prev_y = unvisited.pop(0)
                    for next_x, next_y in get_surround(prev_x, prev_y).values():
                        if map[next_x, next_y] == i and visited[next_x, next_y] == 0:
                            unvisited.append((next_x, next_y))
                            tmp_score += 1
                            visited[next_x, next_y] = 1
                
                if max_connect[i - 1] < tmp_score:
                    max_connect[i - 1] = tmp_score

    for i in range(4):
        score[i] = score[i] * 3 + max_connect[i]
    
    if org_score:
        return score
    
    score = [sorted(score).index(x) for x in score]

    return score

def is_skip(player_id, map, sheep):
    '''
    Confirm if we want to skip this player's turn
    '''
    for x, row in enumerate(sheep):
        for y, n in enumerate(row):
            if n > 1 and player_id == map[x, y]:
                if valid_moves(player_id, map, sheep, x, y):
                    # if there exist at least one valid move return false
                    return False
    return True

def is_done(map, sheep):
    '''
    Check if the game is over
    '''

    for id in range(1, 5):
        if not is_skip(id, map, sheep):
            return False
    return True


def valid_moves(player_id, map, sheep, x, y):
    '''
    Get a vaild moves direction of (x, y) for player i.
    '''

    results = []
    if map[x, y] != player_id or sheep[x, y] <= 1:
        return []
    surround = get_surround(x, y)
    for key, value in surround.items():
        (x_iter, y_iter) = value
        if map[x_iter, y_iter] == 0:
            results.append(key)
    return results

def get_surround(x, y):
    results = {}
    if x - 1 >= 0:
        results[3] = (x - 1, y)
    if x + 1 < 12:
        results[4] = (x + 1, y)
    if y - 1 >= 0:
        if y % 2 == 0:
            if x - 1 >= 0:
                results[1] = (x - 1, y - 1)
            results[2] = (x, y - 1)
        else:
            results[1] = (x, y - 1)
            if x + 1 < 12:
                results[2] = (x + 1, y - 1)
    if y + 1 < 12:
        if y % 2 == 0:
            if x - 1 >= 0:
                results[5] = (x - 1, y + 1)
            results[6] = (x, y + 1)
        else:
            results[5] = (x, y + 1)
            if x + 1 < 12:
                results[6] = (x + 1, y + 1)
    return results

def get_next(pos, dir):
    x, y = pos
    if dir == 1:
        if y == 0 or x - 1 + (y % 2) < 0:
            return (None, None)
        return (x - 1 + (y % 2), y - 1)
    elif dir == 2:
        if y == 0 or x + (y % 2) >= 12:
            return (None, None)
        return (x + (y % 2), y - 1)
    elif dir == 3:
        if x == 0:
            return (None, None)
        return (x - 1, y)
    elif dir == 4:
        if x == 11:
            return (None, None)
        return (x + 1, y)
    elif dir == 5:
        if y == 11 or x - 1 + (y % 2) < 0:
            return (None, None)
        return (x - 1 + (y % 2), y + 1)
    elif dir == 6:
        if y == 11 or x + (y % 2) >= 12:
            return (None, None)
        return (x + (y % 2), y + 1)

class State:
    def __init__(self, id, map, sheep):
        self.id = id
        self.map = np.array(map)
        self.sheep = np.array(sheep)
        self.round = 1
        self.action_list = []
        self.init = self.is_init()

    def reset(self):
        self.map = np.zeros((12, 12))

    def is_init(self):
        for i in range(1, 5):
            if i not in self.map:
                return False
        return True
    
    def init_state(self, id, pos):
        x, y = pos
        self.map[x, y] = id
        self.sheep[x, y] = 16

    def step(self, action):
        # TODO: may need validate id
        (org_x, org_y), m, dir = action
        (x, y) = get_next((org_x, org_y), dir)
        while (x, y) != (None, None) and self.map[x, y] == 0:
            (dst_x, dst_y) = (x, y)
            (x, y) = get_next((x, y), dir)
            # print('roll step')

        next_state = State(self.id % 4 + 1, self.map, self.sheep)
        next_state.round = self.round + 1
        next_state.map[dst_x, dst_y] = self.id
        next_state.sheep[dst_x, dst_y] = m
        next_state.sheep[org_x, org_y] -= m
        next_state.action_list = self.action_list + [action]

        return next_state, next_state.get_score(), False, None

    def get_random_next(self):
        
        
        
        # if the state is not init
        tbd = []
        if self.id not in self.map:
            # for x in range(12):
            #     for y in range(12):
                    # if self.map[x, y] == 0:
            z_map = np.argwhere(self.map == 0) # this
            for pos in z_map:
                x, y = pos
                surround = get_surround(x, y).values()    
                if len(surround) != 6:
                    tbd.append((x, y))
                else:
                    for pos in surround:
                        if self.map[pos] == -1:
                            tbd.append((x, y))
                            break
            x, y = random.choice(tbd)
            next_state = State(self.id % 4 + 1, self.map, self.sheep)
            next_state.init_state(self.id, (x, y))
            next_state.action_list = self.action_list + [((x, y), 16, 0)]
            next_state.round = self.round + 1
            return next_state

        tbd = []
        
        # for x, row in enumerate(self.sheep):
        #     for y, n in enumerate(row):
        sheep_more_than_one = np.argwhere(self.sheep > 1)
        for pos in sheep_more_than_one:
            x, y = pos
            if self.id == self.map[x, y] and valid_moves(self.id, self.map, self.sheep, x, y) != []:
                tbd.append((x, y))
        
        if tbd == []:
            next_id = self.id % 4 + 1
            skip_state = State(next_id, self.map, self.sheep)
            skip_state.round = self.round + 1
            skip_state.action_list = self.action_list
            return skip_state

        dir_list = []
        max_m = 0
        while dir_list == [] or max_m < 1:
            (x, y) = random.choice(tbd)
            max_m = self.sheep[x, y]
            dir_list = valid_moves(self.id, self.map, self.sheep, x, y)
            # print('row_dir')
        
        dir = random.choice(dir_list)
        m = random.randint(1, max_m - 1)
        next_state, _, _, _ = self.step(((x, y), m, dir))
        return next_state                        

    def get_score(self):
        score = [0, 0, 0, 0]
        
        return score

### for MCTS tree ###

class Node:
    def __init__(self, state: State, parent=None):
        self.id = state.id
        self.state = state
        self.map = state.map
        self.sheep = state.sheep
        self.parent = parent
        self.children = []
        self.visit_times = 0
        self.q_value = [0, 0, 0, 0]

    def is_expanded(self):

        tbd = []
        if self.id not in self.map:
            # for x in range(12):
            #     for y in range(12):
                    # if self.map[x, y] == 0:
            z_map = np.argwhere(self.map == 0) # this
            for pos in z_map:
                x, y = pos
                surround = get_surround(x, y).values()    
                if len(surround) != 6:
                    tbd.append((x, y))
                else:
                    for pos in surround:
                        if self.map[pos] == -1:
                            tbd.append((x, y))
                            break
            if len(self.children) == len(tbd):
                return True
            else:
                return False
                    

        count_max = 1 # skip state
        # for x in range(12):
        #     for y in range(12):
        for pos in np.argwhere(self.sheep > 1):
            x, y = pos
            if self.sheep[x, y] > 1 and self.map[x, y] == self.id:
                count_max += len(valid_moves(self.id, self.map, self.sheep, x, y)) * (self.sheep[x, y] - 1)
        if len(self.children) == count_max:
            return True
        else:
            return False

    def best_child(self, is_exploration=False):
        '''
        Using the UCB algorithm, the child node with the highest score 
        is selected after weighing exploration and exploitation, 
        and the one with the highest current Q value is 
        selected directly if it is the prediction stage.
        '''
        best_score = -2147483648
        best_sub_node = None


        for sub_node in self.children:
            if is_exploration:
                C = 1 / 3 ** 0.5
            else:
                C = 0.0
            
            expected = sub_node.q_value[self.id - 1] / sub_node.visit_times
            uncertainty = 2 * math.log(self.visit_times) / sub_node.visit_times
            score = expected + C * uncertainty ** 0.5

            if score > best_score:
                best_sub_node = sub_node
                best_score = score

        if best_sub_node == None:
            skip_node = Node(self.state, self)
            skip_node.id = self.id % 4 + 1
            self.children.append(skip_node)

        return best_sub_node

    def expend(self):
        expended = [ child.state for child in self.children ]
        new_state = self.state.get_random_next()
        while new_state in expended:
            new_state = self.state.get_random_next()

        sub_node = Node(new_state, self)
        self.children.append(sub_node)
        return sub_node


def tree_policy(node: Node):
    while is_done(node.map, node.sheep) == False or node.state.id not in node.state.map:
        

        
        if node.is_expanded():
            node = node.best_child(True)
        else:
            sub_node = node.expend()
            return sub_node

    return node

def default_policy(node: Node):
    current_state = node.state
    while is_done(current_state.map, current_state.sheep) == False or current_state.id not in current_state.map:
        # choose a random action
        current_state = current_state.get_random_next()
    
    final_state_reward = get_reward(current_state.map)
    return final_state_reward

def backup(node: Node, reward):
    while node != None:
        node.visit_times += 1
        for i in range(4):
            node.q_value[i] += reward[i]

        node = node.parent

def mcts(node: Node, computation_budget=1.5):
    start_time = process_time()    
    while within_computational_budget(start_time, computation_budget):
        expend_node = tree_policy(node)

        reward = default_policy(expend_node)

        backup(expend_node, reward)

    best_next_node = node.best_child(False)
    return best_next_node

def within_computational_budget(start, computation_budget=1.5):
    """
    Returns True if time still hasn't run out for the computer's turn.
    """
    elapsed_time = process_time() - start
    return elapsed_time < computation_budget



def render(current_node: Node, k):
    bound = 20
    rec_side = 48
    
    img = np.zeros((rec_side * 13 + 4 * bound + 48, int(rec_side * 12.5) + 2 * bound , 3))


    color = [(255, 255, 255), (0, 255, 255), (153, 255, 204), (255, 204, 255), (255, 255, 153), (160, 160, 160)]

    for i in range(12):
        for j in range(12):
            n = int(current_node.state.sheep[j, i])
            id = int(current_node.state.map[j, i])
            cv2.rectangle(img, (j * 48 + 24 * (i % 2) + bound, i * 48 + 2 * bound), (j * 48 + 24 * (i % 2) + 46 + bound, i * 48 + 46 + 2 * bound), color[id], thickness=-1)
            # cv2.circle(img, (j * 48 + 24 * (i % 2) + 24 + 5, i * 46 + 24 + 5), 24, color[id], thickness=-1)
            if n > 0:
                cv2.putText(img, str(n), (j * 48 + 24 * (i % 2) + 4 + bound, i * 48 + 30 + 5 + 2 * bound), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.rectangle(img, (bound, int(rec_side * 12) + 4 * bound), (bound + int(rec_side * 12.5), int(rec_side * 13.5) + 4 * bound), (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (bound + int(rec_side * 6.5), int(rec_side * 12.5) + 4 * bound), (bound + rec_side * 7, rec_side * 13 + 4 * bound), color[int(current_node.state.id)], thickness=-1)
    cv2.putText(img, 'round: ' + str(current_node.state.round) , (bound , int(rec_side * 13) + 4 * bound), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'Player: ' + str(current_node.state.id) , (bound + rec_side * 4, int(rec_side * 13) + 4 * bound), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    score = get_reward(current_node.state.map, True)

    for i in range(4):
        cv2.rectangle(img, ((i + 8) * 48 + 24  + bound, 13 * 48 + 2 * bound), ((i + 8) * 48 + 24 + 46 + bound, 13 * 48 + 46 + 2 * bound), color[i + 1], thickness=-1)
        cv2.putText(img, str(score[i]), ((8 + i) * 48 + 24 + 4 + bound, 13 * 48 + 30 + 5 + 2 * bound), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('render/' + str(k) + '.jpg', img)

def make_gif(k):
    images = []
    for i in range(k + 1):
        images.append(imageio.imread('./render/' + str(i) + '.jpg'))
    imageio.mimsave('./animate.gif', images)



# py_files = glob.glob('./render/*.jpg')

# for py_file in py_files:
#     os.remove(py_file)

# # Create the initialized state and initialized node
# init_state = State(1, map_lu, sheep_0)
# init_node = Node(init_state)
# current_node = init_node

# for k in range(100):
#     render(current_node)
#     # print("Play round: {}, Player: {}".format(k + 1, current_node.id))
    
    
#     next_node = mcts(current_node, 4) # computation budget

#     if next_node:
#         print(" Choose action: {}".format(next_node.state.action_list[-1]))
#         current_node = next_node
#     else:
#         break
# score = get_reward(current_node.state.map, True)
# print('Yellow:', score[0], ', Green:', score[1], ', Pink:', score[2], ', Blue:', score[3])

# make_gif(k)




# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()


init_pos = InitPos(mapStat, playerID)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)

