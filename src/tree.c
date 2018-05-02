#include <stdio.h>
#include <stdlib.h>
#include "tree.h"
#include "utils.h"
#include "data.h"

void change_leaves(tree *t, char *leaf_list)
{
    list *llist = get_paths(leaf_list);
    char **leaves = (char **)list_to_array(llist);
    int n = llist->size;
    int i,j;
    int found = 0;
    for(i = 0; i < t->n; ++i){
        t->leaf[i] = 0;
        for(j = 0; j < n; ++j){
            if (0==strcmp(t->name[i], leaves[j])){
                t->leaf[i] = 1;
                ++found;
                break;
            }
        }
    }
    fprintf(stderr, "Found %d leaves.\n", found);
}

float get_hierarchy_probability(float *x, tree *hier, int c, int stride)
{
    float p = 1;
    while(c >= 0){
        p = p * x[c*stride];
        c = hier->parent[c];
    }
    return p;
}

/*
计算层级概率
*/
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j*stride] *= predictions[parent*stride]; 
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

/*
寻找一个预测输出中预测概率最高的节点在树中的位置
*/
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while(1){
        float max = 0;
        int max_i = 0;

        //一个组中寻找概率最大的节点
        for(i = 0; i < hier->group_size[group]; ++i){
            int index = i + hier->group_offset[group];
            float val = predictions[(i + hier->group_offset[group])*stride];
            if(val > max){
                max_i = index;
                max = val;
            }
        }
        /*如果概率最大节点的概率大于阈值的话，则在子节点中寻找最大概率,
        此处的阈值并不是-thresh的阈值，而是-hier的阈值,
        为什么要设置两个阈值，目前还没想明白？？？？猜测是为了控制
        要满足多大的阈值才允许选择该节点下的子节点
        */
        if(p*max > thresh){
            p = p*max;
            group = hier->child[max_i];
            //没有孩子节点的话就返回当前节点
            if(hier->child[max_i] < 0) return max_i;
        } else if (group == 0){
            //group==0为根节点
            return max_i;
        } else {
            //如果该节点的概率较低且该节点不是根节点，则返回父节点的位置
            //此处为什么不使用以下简单的代码：return hier->parent[max];
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

tree *read_tree(char *filename)
{
    tree t = {0};
    FILE *fp = fopen(filename, "r");

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while((line=fgetl(fp)) != 0){
        char *id = calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = realloc(t.parent, (n+1)*sizeof(int));
        t.parent[n] = parent;

        t.child = realloc(t.child, (n+1)*sizeof(int));
        t.child[n] = -1;

        t.name = realloc(t.name, (n+1)*sizeof(char *));
        t.name[n] = id;
        if(parent != last_parent){
            ++groups;
            t.group_offset = realloc(t.group_offset, groups * sizeof(int));
            t.group_offset[groups - 1] = n - group_size;
            t.group_size = realloc(t.group_size, groups * sizeof(int));
            t.group_size[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = realloc(t.group, (n+1)*sizeof(int));
        t.group[n] = groups;
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    ++groups;
    t.group_offset = realloc(t.group_offset, groups * sizeof(int));
    t.group_offset[groups - 1] = n - group_size;
    t.group_size = realloc(t.group_size, groups * sizeof(int));
    t.group_size[groups - 1] = group_size;
    t.n = n;
    t.groups = groups;
    t.leaf = calloc(n, sizeof(int));
    int i;
    for(i = 0; i < n; ++i) t.leaf[i] = 1;
    for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

    fclose(fp);
    tree *tree_ptr = calloc(1, sizeof(tree));
    *tree_ptr = t;
    //error(0);
    return tree_ptr;
}
