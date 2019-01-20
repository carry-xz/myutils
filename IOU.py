# -*- coding: utf8 -*-
# auther:carry-xz
# 20190120

import time
import numpy as np 
'''
实现任意的两个多边形（单连通域）之间的IOU计算。此处考虑中心点位置因素，不是yolo中的IOU
取得多边形所有像素作为集合，然后求交集并集。
'''
def _linspace(start,end,delt=1):
    if start==end or delt==0:
        return []
    res = [start]
    if start<=end:
        assert delt>=0 , 'delt should >= 0'
    else:
        if delt==1:
            delt = -1
        assert delt<0 , 'delt should < 0'
    if delt>0:
        while start+delt<end:
            start += delt
            res.append(start)
    if delt<0:
        while start+delt>end:
            start += delt
            res.append(start)
    if end not in res:
        res.append(end)
    return res

def gradientCalc(point1,point2):
    assert point1!=point2,'calc gradient should between different points'
    dx = point2[0]-point1[0]
    dy = point2[1]-point1[1]
    if dx==0:
        if point1[1]<point2[1]:
            grd = float("inf")
        else:
            grd = -float("inf")
    else:
        grd = dy/dx
    return grd

def getPointsBetweenTowPoint(point1,point2):
    # point:tuple:(int x,int y) 
    # 获取两个点之间所有像素点,存入集合中（便于查询）
    res = set()
    grd = gradientCalc(point1,point2)
    if abs(grd)>1:
        ind_list = _linspace(point1[1],point2[1])
        # ind_list = ind_list[:-1] #左闭右开
        x0,y0 = point1[0],point1[1]
        dx = point1[0]-point2[0]
        dy = point1[1]-point2[1]
        for y in ind_list:
            x = x0+round((y-y0)*dx/dy)
            res.add((x,y))
    else:
        ind_list = _linspace(point1[0],point2[0])        
        # ind_list = ind_list[:-1] #左闭右开
        x0,y0 = point1[0],point1[1]
        dx = point1[0]-point2[0]
        dy = point1[1]-point2[1]
        for x in ind_list:
            y = y0+round((x-x0)*dy/dx)
            res.add((x,y))
    return res

def position2index(point1,xlimit,ylimit):
    # point:tuple:(int x,int y) 
    # 将点坐标转化为索引，索引计算方式y*行数+x
    return point1[1]*xlimit+point1[0]

def index2position(index,xlimit,ylimit):
    # 将索引转化为点坐标，索引计算方式y*行数+x
    x = index//xlimit
    y = index%xlimit
    return (int(x),int(y))

def checkPointlineCrossLine(point1,LinePoint1,LinePoint2):
    # 射线法：判断点所在的水平向右射线是否与线段相交，线段由两个点表示
    # 在线上的点不认为在内部。
    y = point1[1]
    if LinePoint1[1]<y and LinePoint2[1]<y:
        return False
    elif LinePoint1[1]>y and LinePoint2[1]>y:
        return False
    else:
        point_set = getPointsBetweenTowPoint(LinePoint1,LinePoint2)
        for point in point_set:#可以优化
            if point[1]== y:
                if point[0]>=point1[0]:
                    return True
        return False

def checkPointInPolygon(point1,polygon):
    # point:tuple:(int x,int y)
    # polygon:List:[point1,point2,...]
    # 查看点是否在多边形内部
    count = 0
    for i in range(len(polygon)-1):
        if checkPointlineCrossLine(point1,polygon[i],polygon[i+1]):
            count +=1
    if checkPointlineCrossLine(point1,polygon[-1],polygon[0]):
        count +=1
    if count%2==0:
        return False
    else:
        return True
    
def getNeighbourPoint(point):
    x,y = point
    return {(x,y+1),(x,y-1),(x+1,y),(x-1,y)}


def getAllPointOnPolygonLine(polygon):
    # polygon:List:[point1,point2,...]
    # out:set:{p1,p2,...}
    # 获取多边形边线上所有点，存入结果集合中。
    assert len(polygon)>=3,'polygon should have at least 3 point'
    line_point_set = set()
    for i in range(len(polygon)-1):
        temp_set = getPointsBetweenTowPoint(polygon[i],polygon[i+1])
        line_point_set = line_point_set.union(temp_set)
    temp_set = getPointsBetweenTowPoint(polygon[-1],polygon[0])
    line_point_set = line_point_set.union(temp_set)
    return line_point_set 

def getAllPointInPolygon(polygon):
    # polygon:List:[point1,point2,...]
    # 取第一个顶点的斜上角点或斜下角点作为起点
    # 通过多边形内部一个点获取多边形内部所有点,内部点不包括边线上的点
    
    st_point=None
    line_set = getAllPointOnPolygonLine(polygon)
    for point1 in polygon:
        x,y = point1
        temp_points = [(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)]
        for t_point in temp_points:
            if t_point in line_set:
                continue
            if checkPointInPolygon(t_point,polygon):
                st_point = t_point
                break

    open_set=set()
    close_set=set()
    res_set=set()
    if st_point is not None:
        open_set.add(st_point)
        res_set.add(st_point)
        print('start point:',st_point)
    while open_set:
        curpoint = open_set.pop()
        close_set.add(curpoint)
        curList = getNeighbourPoint(curpoint)
        for tpoint in curList:
            if tpoint not in line_set:
                if tpoint not in close_set:
                    open_set.add(tpoint)
                    res_set.add(tpoint)
                    # print('res_set add',tpoint)
        # print('open_set',open_set)
        # print('close_set',close_set)
    return res_set.union(line_set)

def recuriseAddPoint(cur_point,close_set,res_set,line_set):
    # 递归方式添加多边形内部点
    p_list = getNeighbourPoint(cur_point)
    for point in p_list:
        if point in close_set:
            continue
        else:
            close_set.add(point)
        if point not in line_set :
            res_set.add(point)
            recuriseAddPoint(point,close_set,res_set,line_set)

def iouCalc(polygon1,polygon2):
    # point:tuple:(int x,int y)
    # polygon:List:[point1,point2,...]
    # 计算多边形IOU
    pointset1 = getAllPointInPolygon(polygon1)
    pointset2 = getAllPointInPolygon(polygon2)
    inters = len(pointset1.intersection(pointset2))
    unio = len(pointset1.union(pointset2))
    return inters/unio

def displayPolygon(pointSet):
    # 用于显示结果
    pass 


import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # print(_linspace(10,20,7))
    print('start............')
    # polygon=[(0,0),(5,0),(6,6),(-2,5)]
    # polygon2=[(0,0),(3,0),(4,3),(0,3)]
    polygon=[(0,0),(100,0),(100,100),(0,100)]
    polygon2=[(0,0),(30,0),(30,30),(0,30)]
    print(checkPointInPolygon((-1,6),polygon))
    res = getAllPointOnPolygonLine(polygon)
    # res = getPointsBetweenTowPoint((1,1),(5,7))
    # print(checkPointInPolygon((-1,0),[(0,0),(5,0),(6,6),(-2,5)]))
    res = getAllPointInPolygon(polygon)
    t1 = time.time()
    iou = iouCalc(polygon,polygon2)
    print('iou:',iou)
    print('cost time:',time.time()-t1)

    x = [item[0] for item in res]
    y = [item[1] for item in res]
    plt.scatter(x,y)
    plt.show()
