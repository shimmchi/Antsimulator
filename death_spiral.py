import numpy as np
import matplotlib.pyplot as plt
import random

class Plane:
    def __init__(self):
        self.MAX_X = 100
        self.MIN_X = 0
        self.MAX_Y = 100
        self.MIN_Y = 0


# 進行方向クラス
class Direction:
    def validator(self, x, y):
        valid_x = x
        valid_y = y
        if x < 0:
            valid_x = self.X_DICT['left']
        if x == 0:
            valid_x = self.X_DICT['center']
        if x > 0:
            valid_x = self.X_DICT['right']

        if y < 0:
            valid_y = self.Y_DICT['down']
        if y == 0:
            valid_y = self.Y_DICT['center']
        if y > 0:
            valid_y = self.Y_DICT['up']
        return valid_x, valid_y

    def __init__(self):
        self.X_DICT = {'left': -1, 'center': 0, 'right': 1}
        self.Y_DICT = {'down': -1, 'center': 0, 'up': 1}


    def rotate(self, x, y, sita):
        R = np.array([[np.cos(sita), -1.*np.sin(sita)],[np.sin(sita), np.cos(sita)]])
        before = np.array([x, y])
        after = R.dot(before)
        return self.validator(after[0], after[1])

    def add(self, x1, y1, x2, y2):
        return self.validator(x1+x2, y1+y2)






###############
# environment #
###############

class Feed:
    def validator(self, amount):
        valid_amount = amount
        if amount < self.MIN_AMOUNT:
            valid_amount = self.MIN_AMOUNT
        return valid_amount

    def __init__(self):
        self.MIN_AMOUNT = 0

    def isFeed(self, amount):
        return amount > self.MIN_AMOUNT


class Pheromones:
    def validator(self, amount, status):
        valid_amount = amount
        valid_status = status
        if amount < self.MIN_AMOUNT:
            valid_amount = self.MIN_AMOUNT
        if amount > self.MAX_AMOUNT:
            valid_amount = self.MAX_AMOUNT
        if status >= 3:
            valid_status = 3
        if status >=2 and status < 3:
            valid_status = 2
        if status >= 1 and status < 2:
            valid_status = 1
        if status < 1:
            valid_status = 0
        return valid_amount, valid_status

    def __init__(self):
        self.MIN_AMOUNT = 1
        self.MAX_AMOUNT = 300
        self.SPREAD_ATTEN_RATE = 0.3
        self.EVAPOLATE_RATE = 0.9
        self.AMOUNT_L1 = self.MAX_AMOUNT*0.1
        self.AMOUNT_L2 = self.MAX_AMOUNT*0.5
        self.AMOUNT_L3 = self.MAX_AMOUNT*0.9
        self.SPREAD_DICT = {'start': 3, 'first': 2, 'second': 1, 'none': 0}

    def isPheromones(self, amount):
        return amount > self.MIN_AMOUNT

    def isPheromonesL1(self, amount):
        return amount > self.MIN_AMOUNT and amount <= self.AMOUNT_L1

    def isPheromonesL2(self, amount):
        return amount > self.AMOUNT_L1 and amount <= self.AMOUNT_L2

    def isPheromonesL3(self, amount):
        return amount > self.AMOUNT_L2 and amount <= self.AMOUNT_L3

    def isPheromonesL4(self, amount):
        return amount > self.AMOUNT_L3

    def evapolate(self, amount, status):
        return self.validator(amount*self.EVAPOLATE_RATE, status - 1)

    def spread(self, amount, status):
        if status == self.SPREAD_DICT['first']:
            return amount*self.SPREAD_ATTEN_RATE
        if status == self.SPREAD_DICT['second']:
            return amount*self.SPREAD_ATTEN_RATE*self.SPREAD_ATTEN_RATE
        return 0


class Ground(Plane):
    def __init__(self):
        super().__init__()
        self.PHEROMONES = Pheromones()
        self.FEED = Feed()

        # それぞれのマスにはエサとフェロモンが配置される可能性がある
        self.np_feed_amount       = np.zeros((self.MAX_X+1, self.MAX_Y+1))
        self.np_pheromones_amount = np.zeros((self.MAX_X+1, self.MAX_Y+1))
        self.np_pheromones_status = np.zeros((self.MAX_X+1, self.MAX_Y+1))

    def validateCoordinate(self, x, y):
        valid_x = x
        valid_y = y
        if x > self.MAX_X:
            valid_x = self.MAX_X
        if x < self.MIN_X:
            valid_x = self.MIN_X
        if y > self.MAX_Y:
            valid_y = self.MAX_Y
        if y < self.MIN_Y:
            valid_y = self.MIN_Y
        return valid_x, valid_y


    def setFeed(self, x, y, amount):
        init_amount = self.np_feed_amount[x][y]
        sum = self.FEED.validator(init_amount + amount)
        self.np_feed_amount[x][y] = sum

    def decrementFeed(self, x, y):
        init_amount = self.np_feed_amount[x][y]
        decremented = self.FEED.validator(init_amount - 1)
        self.np_feed_amount[x][y] = decremented

    def isFeed(self, x, y):
        amount = self.np_feed_amount[x][y]
        return self.FEED.isFeed(amount)

    def addPheromones(self, x, y, amount, status):
        init_amount = self.np_pheromones_amount[x][y]
        n_amount, n_status = self.PHEROMONES.validator(init_amount + amount, status)
        self.np_pheromones_amount[x][y] = n_amount
        self.np_pheromones_status[x][y] = n_status

    def isPheromones(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        return self.PHEROMONES.isPheromones(amount)

    def isPheromonesL1(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        return self.PHEROMONES.isPheromonesL1(amount)

    def isPheromonesL2(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        return self.PHEROMONES.isPheromonesL2(amount)

    def isPheromonesL3(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        return self.PHEROMONES.isPheromonesL3(amount)

    def isPheromonesL4(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        return self.PHEROMONES.isPheromonesL4(amount)

    def evapolatePheromones(self, x, y):
        init_amount = self.np_pheromones_amount[x][y]
        init_status = self.np_pheromones_status[x][y]
        evapolated_amount, evapolated_status = self.PHEROMONES.evapolate(init_amount, init_status)
        self.np_pheromones_amount[x][y] = evapolated_amount
        self.np_pheromones_status[x][y] = evapolated_status

    def spreadPheromones(self, x, y):
        amount = self.np_pheromones_amount[x][y]
        status = self.np_pheromones_status[x][y]
        if status == self.PHEROMONES.SPREAD_DICT['none'] or status == self.PHEROMONES.SPREAD_DICT['start']:
            return

        spread_amount = self.PHEROMONES.spread(amount, status)
        target = []
        if status == self.PHEROMONES.SPREAD_DICT['first']:
            target = [[x+1,y], [x-1,y], [x,y+1], [x,y-1]]

        if status == self.PHEROMONES.SPREAD_DICT['second']:
            target = [[x+2,y], [x+1,y+1], [x,y+2], [x-1,y+1], [x-2,y], [x-1,y-1], [x, y-2], [x+1, y-1]]

        for point in target:
            if point[0] == x and point[1] == y:
                continue
            point_x, point_y = self.validateCoordinate(point[0], point[1])
            self.addPheromones(point_x, point_y, spread_amount, self.PHEROMONES.SPREAD_DICT['none'])


    def update(self):
        for x in range(self.MAX_X):
            for y in range(self.MAX_Y):
                if self.isPheromones(x, y):
                    self.spreadPheromones(x, y)
                    self.evapolatePheromones(x, y)


    def draw(self, sub_plot):
        feed          = np.array([[ self.isFeed(x, y) for x in range(self.MAX_X)] for y in range(self.MAX_Y) ])
        pheromones_l1 = np.array([[ self.isPheromonesL1(x, y)  for x in range(self.MAX_X) ] for y in range(self.MAX_Y) ])
        pheromones_l2 = np.array([[ self.isPheromonesL2(x, y)  for x in range(self.MAX_X) ] for y in range(self.MAX_Y) ])
        pheromones_l3 = np.array([[ self.isPheromonesL3(x, y)  for x in range(self.MAX_X) ] for y in range(self.MAX_Y) ])
        pheromones_l4 = np.array([[ self.isPheromonesL4(x, y)  for x in range(self.MAX_X) ] for y in range(self.MAX_Y) ])

        sub_plot.scatter(np.where(feed)[0], np.where(feed)[1], s=40, c='green',marker='x')
        sub_plot.scatter(np.where(pheromones_l1 )[0], np.where(pheromones_l1 )[1], s=40, c='red', alpha=0.1, marker='o')
        sub_plot.scatter(np.where(pheromones_l2 )[0], np.where(pheromones_l2 )[1], s=40, c='red', alpha=0.4, marker='o')
        sub_plot.scatter(np.where(pheromones_l3 )[0], np.where(pheromones_l3 )[1], s=40, c='red', alpha=0.6, marker='o')
        sub_plot.scatter(np.where(pheromones_l4 )[0], np.where(pheromones_l4 )[1], s=40, c='red', alpha=1.0, marker='o')



############################################


################
# ant & colony #
################

class AntColony:
    def validator(self):
        if self.ANT_NUM < 0:
            self.ANT_NUM = 0
        self.NEST_X, self.NEST_Y = self.GND.validateCoordinate(self.NEST_X, self.NEST_Y)

    def __init__(self, ant_num, nest_x, nest_y):
        self.ANT_NUM = ant_num
        self.NEST_X = nest_x
        self.NEST_Y = nest_y
        self.GND = Ground()
        self.DIRECTION = Direction()
        self.validator()

        # アリが進行方向を変える確率
        self.PROB_CHANGE_DIRECTION = 0.3

        # アリが残すフェロモン
        self.ANT_PHEROMONES_AMOUNT = 100
        self.ANT_PHEROMONES_STATUS = 3

        self.ANT_STATE = {'randomWalk':0, 'goToNest': 1}

        self.np_ant_position  = np.array(( [[self.NEST_X, self.NEST_Y]]*self.ANT_NUM ))
        self.np_ant_direction = np.random.randint(-1, 2, (self.ANT_NUM, 2))
        self.np_ant_state     = np.zeros(self.ANT_NUM)

    def isNest(self, x, y):
        return x == self.NEST_X and y == self.NEST_Y

    def setFeed(self, x, y, amount):
        self.GND.setFeed(x, y, amount)

    def walk(self, n):
        x  = self.np_ant_position[n][0]
        y  = self.np_ant_position[n][1]
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        n_x, n_y = self.GND.validateCoordinate(x + dx, y + dy)
        self.np_ant_position[n][0] = n_x
        self.np_ant_position[n][1] = n_y
        return

    def randomChangeDirection(self, n):
        if random.random() > self.PROB_CHANGE_DIRECTION:
            return
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        rand_dx = random.randint(-1,1)
        rand_dy = random.randint(-1,1)
        n_dx, n_dy = self.DIRECTION.validator(dx+rand_dx, dy+rand_dy)
        self.np_ant_direction[n][0] = n_dx
        self.np_ant_direction[n][1] = n_dy
        return

    def changeDirectionToNest(self, n):
        x  = self.np_ant_position[n][0]
        y  = self.np_ant_position[n][1]
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        dx_to_nest, dy_to_nest = self.DIRECTION.validator(self.NEST_X - x, self.NEST_Y - y)
        n_dx = random.choice([dx, dx_to_nest])
        n_dy = random.choice([dy, dy_to_nest])

        self.np_ant_direction[n][0] = n_dx
        self.np_ant_direction[n][1] = n_dy
        return

    def searchFeed(self, n):
        x  = self.np_ant_position[n][0]
        y  = self.np_ant_position[n][1]
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        n_dx = dx
        n_dy = dy
        for i in range(-1,2):
            for j in range(-1,2):
                n_x, n_y = self.GND.validateCoordinate(x + i, y + j)
                if self.GND.isFeed(n_x, n_y):
                    n_dx = i
                    n_dy = j
        self.np_ant_direction[n][0] = n_dx
        self.np_ant_direction[n][1] = n_dy
        return

    def pheromonesTrail(self, n):
        x  = self.np_ant_position[n][0]
        y  = self.np_ant_position[n][1]
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        found_pheromones = False
        sita = np.pi*0.25
        LEFT_RIGHT = [-1,1]
        looking_direction = random.randint(0,1)

        # 真後ろは探索しない
        for angle in range(0,4):
            if found_pheromones:
                continue
            for i in range(2):
                L_R = LEFT_RIGHT[looking_direction]
                n_dx, n_dy = self.DIRECTION.rotate(dx, dy, L_R*angle*sita)
                l_x, l_y = self.GND.validateCoordinate(x + n_dx, y + n_dy)
                if self.GND.isPheromones(l_x, l_y):
                    self.np_ant_direction[n][0] = n_dx
                    self.np_ant_direction[n][1] = n_dy
                    found_pheromones = True
                looking_direction = (looking_direction + 1)%2


    # 何らかの理由で巣の位置を誤認する
    # 誤認する巣の位置は、現在の位置を中心として、
    # 正しい巣の位置を一定角度回転した位置と仮定する
    def changeDirectionToFalseNest(self, n):
        x  = self.np_ant_position[n][0]
        y  = self.np_ant_position[n][1]
        dx = self.np_ant_direction[n][0]
        dy = self.np_ant_direction[n][1]
        pos  = np.array([x, y])
        nest = np.array([self.NEST_X, self.NEST_Y])
        sita = np.pi*0.49
        R=np.array([[np.cos(sita),-1.*np.sin(sita)],[np.sin(sita),np.cos(sita)]])
        false_nest = R.dot(nest-pos) + pos

        dx_to_nest, dy_to_nest = self.DIRECTION.validator(false_nest[0]-x, false_nest[1]-y)

        n_dx = random.choice([dx, dx_to_nest])
        n_dy = random.choice([dy, dy_to_nest])
        self.np_ant_direction[n][0] = n_dx
        self.np_ant_direction[n][1] = n_dy
        return



    def updateGND(self):
        self.GND.update()

    def update(self):
        for n in range(self.ANT_NUM):
            x  = self.np_ant_position[n][0]
            y  = self.np_ant_position[n][1]

            # エサを見つけていない時
            if self.np_ant_state[n] == self.ANT_STATE['randomWalk']:
                # フェロモンを見つけたら
                if self.GND.isPheromones(x, y):
                    # フェロモンを追いかける(フェロモントレイル)
                    self.pheromonesTrail(n)
                    # ランダムに方向を変える
                    self.randomChangeDirection(n)

                # フェロモンが無い時
                else:
                    # ランダムに方向を変える
                    self.randomChangeDirection(n)
                    # 周囲にエサがないか探す
                    self.searchFeed(n)

            # エサを持っている時、巣へ変える
            if self.np_ant_state[n] == self.ANT_STATE['goToNest']:
                # 巣の方向を錯覚してしまう
                self.changeDirectionToFalseNest(n)
                # フェロモンを残す
                self.GND.addPheromones(x, y, self.ANT_PHEROMONES_AMOUNT, self.ANT_PHEROMONES_STATUS)
                # たまに関係のない方向にもいくかも知れない
                self.randomChangeDirection(n)

            # 歩く
            self.walk(n)

            n_x  = self.np_ant_position[n][0]
            n_y  = self.np_ant_position[n][1]

            # エサを見つけたら
            if self.GND.isFeed(n_x, n_y):
                # 巣へ帰る
                self.np_ant_state[n] = self.ANT_STATE['goToNest']
                # エサが１つ減る
                self.GND.decrementFeed(n_x, n_y)

            # 巣に帰ったら再びエサを探し出す
            if self.isNest(n_x, n_y):
                self.np_ant_state[n] = self.ANT_STATE['randomWalk']


    def draw(self, sub_plot):
        # フェロモン・エサを描画する
        self.GND.draw(sub_plot)

        # アリの巣を描画する
        sub_plot.scatter(self.NEST_X,  self.NEST_Y, s=50,c='blue',alpha=0.5,marker='o')

        # 全てのアリを描画する
        sub_plot.scatter(self.np_ant_position.T[0], self.np_ant_position.T[1],s=20,c='blue',marker='o')





##############################################

class Presentation:
    def __init__(self):
        plt.ion()
        plt.axis('equal')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.plane = Plane()

    def draw(self, item):
        self.ax.clear()
        plt.xlim([self.plane.MIN_X, self.plane.MAX_X])
        plt.ylim([self.plane.MIN_Y, self.plane.MAX_Y])

        item.draw(self.ax)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.000001)




###############################################

class Application:
    def __init__(self):
        self.PRESENTATION = Presentation()

        self.ant_colony = AntColony(ant_num = 100, nest_x = 50, nest_y = 50)

        # エサの配置
        self.ant_colony.setFeed(x = 40, y = 10, amount = 30)
        self.ant_colony.setFeed(x = 70, y = 60, amount = 30)

    def run(self):
        while True:
            self.ant_colony.update()
            self.ant_colony.updateGND()
            self.PRESENTATION.draw(self.ant_colony)


################################################

#######
# 実行 #
#######

app = Application()
app.run()
