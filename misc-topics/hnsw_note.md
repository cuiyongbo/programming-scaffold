# insight <abbr title="Hierarchical Navigable Small World">HNSW</abbr>


## Basics

- [nmslib/hnswlib](https://github.com/nmslib/hnswlib)
    - refer to README for Basic usage

### Parameters

- refer to [ALGO_PARAMS](https://github.com/nmslib/hnswlib/ALGO_PARAMS.md)
- search parameters
    - ef 检索过程中寻找最相似候选的有限队列大小. 影响检索精度和耗时. 推荐取值范围 300-1500
    - topk 要召回的候选数. topk越大, 检索越耗时
- index construction parameters
    - ef_construction, 含义同 ef, 作用于索引构建过程, 影响构建时间和索引精度. 推荐取值范围 100-500
    - M 非0层候选的最大连接度, 0层的是 2*M. 影响构建时间, 索引精度, 索引体积. 推荐取值范围 16-100
    - num_elements 索引最大候选个数. 因为 hnws 要预分配内存, 候选数超过 num_elements 后, 会无法继续插入新数据 (应该可以绕过去)


### Supported distances:

| Distance         | parameter       | Equation                |
| -------------    |:---------------:| -----------------------:|
|Squared L2        |'l2'             | d = sum((Ai-Bi)^2)      |
|Inner product     |'ip'             | d = 1.0 - sum(Ai\*Bi)   |
|Cosine similarity |'cosine'         | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) - sum(Bi\*Bi))|


## 检索过程

- 根据请求向量, `enterpoint_node_`,  依次从 `max_level` 层到1层找到和请求向量最相似的候选 cand_id (贪心算法, 每层数据实际是 adjacency-list 格式的图)
- 以第一步找到的 cand_id 为起始点, 在0层中扩散查找相似候选, 扩算方式类似 Dijkstra 算法, 维持一个容量有限的有限队列(可通过 `max(ef_, topk)` 参数设置), 找到 `topk` 个候选
- 按照相似度从大到小对候选做排序
- **Note:** 算法提供了 `BaseFilterFunctor`, 即属性过滤能力, 实现方式是边扩散边过滤

```cpp
typedef size_t labeltype; // label 大小. 通常都是64位系统, 大小等于8字节
typedef float dist_t;
// dist 代表请求向量和候选的距离. dist 值越大, 请求向量和候选越不相似. 模版参数, 一般就是 float
struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
        std::pair<dist_t, tableint> const& b) const noexcept {
        return a.first < b.first;
    }
};
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates; // max-heap, 大根堆, 元素是 {dist, cand_id}, 最不相似结果排在top
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set; // max-heap, 大根堆, 元素是 {-dist, cand_id}, 因为对距离取反, 所以最相似结果排在top
std::priority_queue<std::pair<dist_t, labeltype >> result; // min-heap, 小根堆, 最相似的结果排在top
/*
检索过程中这三个 pq 的作用时:
- candidate_set 用来在检索过程存放中间结果, 每次从 candidate_set 取出一个当前和请求向量最相似的候选做扩散
- top_candidates 一方面是存放可能候选结果; 另外一方面结合 candidate_set 来判断检索是否可以终止了(对比两个 pq 的 top, 如果 candidate_set 中和请求向量最相似的结果都比 top_candidates 中最不相似的结果距离大, 就可以终止了)
- result 存放检索结果, 后续逐个遍历它, 就可以到按相似度从大到小排列的候选序列
*/
```


## 构建过程

构建过程和检索过程类似. 此外得到相似候选后, 还需要在待插入数据和候选间建立双向链接


## 工程优化

- 算法调参: m(节点的最大连接度), cef(构建时优先队列的最大长度), sef(类似cef, 用于检索), topk, space(距离类型: l2, ip, cosine), dim
- 内存优化;
    - 使用 __mm_prefetch, 预取数据放入cache line, 减少 cache miss
    - 开启透明大页内存(Transparent Huge Page), 减少 TLB(保存虚拟内存地址到物理地址映射的 cache) Miss 和缺页中断
    - VisitedListPool 内存复用, 减少内存重复申请/释放开销
- 精细的锁粒度保护, 避免卡并发 (保护 max_level, cur_element_count, label_lookup, neighbor_list 精确到节点粒度)
- 支持 embedding 量化(float16, int16, int8等, 在线 query_vec 使用对称量化以加速检索; 构建候选向量使用非对称量化, 精度更高), 指令集加速 (AVX512, AVX, SSE, etc.)
- 补充打点指标: 候选量, 内存占用, 节点的邻居数量分布, 检索遍历的候选数, 过滤掉的候选数, 流式更新情况(添加/更新/删除数量速率, 更新失败), embedding 模值分布
- THP 设置

THP 策略:

```bash
# cat /sys/kernel/mm/transparent_hugepage/enabled
always [madvise] never
# https://man7.org/linux/man-pages/man2/madvise.2.html
```

如何分配释放:

```cpp
// allocate THP
static size_t huge_page_size = 4096 * 1024;  // 4MB
static size_t common_page_size = getpagesize(); // getconf PAGESIZE, usually 4096 -> 4KB
alloc_size = pgnum * huge_page_size; // 按 page_size 向上取整
auto mmap_p = static_cast<char*>(mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0));
int madvret = madvise(arr, alloc_size, MADV_HUGEPAGE);
// free THP
munmap(arr, mem_size);
```

### Int8 量化

Int8 量化有两种量化方式: (QAT和PTQ的区别, 对召回率的影响会降低2-3个点, 召回延时提升一般30%-40%左右(百万候选量级的索引))

- 对每个候选单独计算 vec_max, vec_min, scale 进行量化 (工作中实际在用)
- 所有候选统一计算 vec_max, vec_min, scale 进行量化 

具体的:

- 构建时 int8 量化使用非对称量化, 候选被量化到了 [0, 255], 不使用用对称量化的原因是为了减少精度损失
  - uint8 quantization Data layout: `[dim x uint8], float(scale), float(zero_point), (float(bias))`
- 在线搜索时 int8 量化使用对称量化, query_embedding 被量化到了 [-127, 127], 这么做的原因是对称量化的 zero_point=0, 可以减少在线运算量
  - int8 quantization Data layout: `[dim x int8], float(scale), float(zero_point), (float(bias))`
- 注意开启量化后, 索引embedding数据按照 uint8 格式存储, 计算相似度时也按照 int8/uint8 格式计算, 但结果返回格式还是 float


## 代码注解

```cpp
typedef size_t labeltype; // label 大小. 通常都是64位系统, 大小等于8字节

// https://github.com/nmslib/hnswlib/hnswlib/hnswalg.h
HierarchicalNSW(
    SpaceInterface<dist_t> *s,
    size_t max_elements,
    size_t M = 16,
    size_t ef_construction = 200,
    size_t random_seed = 100,
    bool allow_replace_deleted = false)
    : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
        link_list_locks_(max_elements),
        element_levels_(max_elements),
        allow_replace_deleted_(allow_replace_deleted) {
    max_elements_ = max_elements;
    num_deleted_ = 0;
    data_size_ = s->get_data_size(); // data_size_ = dim - sizeof(float);, embedding 数据大小
    fstdistfunc_ = s->get_dist_func(); // distance type: ip or l2
    dist_func_param_ = s->get_dist_func_param(); // embedding dimension
    maxM_ = M_;
    maxM0_ = M_ - 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = maxM0_ - sizeof(tableint) + sizeof(linklistsizeint); // 0 层 neighbor list 数据大小
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype); // 0层数据大小. 等于 neighbor_list + embedding + label
    offsetData_ = size_links_level0_; // embedding 数据偏移
    label_offset_ = size_links_level0_ + data_size_; // label 数据偏移
    offsetLevel0_ = 0; // 0层数据偏移, 一般索引文件前面也没别的数据, 就是0

    // 0层数据
    // 优化思路: 普通malloc性能较差, 可以结合 mmap+ THP 优化内存读写性能
    // transparent huge page, 也就是透明内存大页, 页大小比普通内存页大, 减少缺页中断, 增加内存页cache命中率
    data_level0_memory_ = (char *) malloc(max_elements_ - size_data_per_element_);
    // 候选数目
    cur_element_count = 0;
    // 检索0层时用来判断一个候选是否已经被计算过了, 类似 dfs/bfs 里面的 color, visited_set
    // 因为检索是个非常频繁的过程, 这里提前预分配 visited_list_pool_, 检索时可以直接使用, 避免重复的内存申请/释放
    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));
    // initializations for special treatment of the first node
    // 只有在 addPoint 过程中才有可能更改. 对于一个构建好的索引,  enterpoint_node_ 来源于索引文件. 是构建时计算好的值.
    // 如果没有流式更新的话, enterpoint_node_ 不变, 对于每次检索是个固定值. 从而确保相同query检索结果是正定的.
    // 但如果有流式更新, 那么 enterpoint_node_ 可能会变, 相同query重复检索, 检索结果也有可能会改变.
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    // 非0层 neighbor list 数据
    linkLists_ = (char **) malloc(sizeof(void *) - max_elements_);
    size_links_per_element_ = maxM_ - sizeof(tableint) + sizeof(linklistsizeint); // 非 0 层的 neighbor list 数据大小
    mult_ = 1 / log(1.0 - M_);
    revSize_ = 1.0 / mult_;
}

// 索引文件大小
size_t indexFileSize() const {
    size_t size = 0;
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(mult_);
    size += sizeof(ef_construction_);

    size += cur_element_count - size_data_per_element_;

    for (size_t i = 0; i < cur_element_count; i++) {
        unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ - element_levels_[i] : 0;
        size += sizeof(linkListSize);
        size += linkListSize;
    }
    return size;
}

// serialize index data to disk
void saveIndex(const std::string &location) {
    // 1. save index header
    writeBinaryPOD(output, offsetLevel0_); // 就是 0, 就是说先保存0层数据
    writeBinaryPOD(output, max_elements_); // 最大能包含多少候选, 一般比 cur_element_count_ 要大, 给流式更新留 buffer
    writeBinaryPOD(output, cur_element_count); // 当前包含的候选数: 各层累加
    writeBinaryPOD(output, size_data_per_element_); // 0层数据大小. 等于 neighbor_list + embedding + label
    writeBinaryPOD(output, label_offset_);  // label 数据的偏移量
    writeBinaryPOD(output, offsetData_);    // embedding 数据的偏移量
    writeBinaryPOD(output, maxlevel_);      // hnsw 包含几张图, 一般不大于 8
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);   // 非0层节点的最大邻居个数
    writeBinaryPOD(output, maxM0_);  // 0层节点的最大邻居个数
    writeBinaryPOD(output, M_);  // 同 maxM_
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);
    // 2. save level 0 data: neighbour_list, embedding, label
    //  0 层数据的存储格式: |linklistsizeint|linkslist|embedding|label|
    output.write(data_level0_memory_, cur_element_count - size_data_per_element_);
    // 3. neighbor_list of a label in non-zero layer: 每层的邻居个数按照 maxM_ 预分配, 当 label 存在于多层时就按层数放大, 这里的思路是保持 label 的 neighbor_list 数据是定长的, 方便存取
    // 非 0 层数据的存储格式: |linklistsizeint|linkslist, 注意它不包括0层的 neighbor list 数据
    for (size_t i = 0; i < cur_element_count; i++) {
        // std::vector<int> element_levels_;  // keeps level of each element
        // element_levels_[i] 表示候选i最高被放到哪一层, 候选插入时, 候选i依次被插入到 element_levels_[i], element_levels_[i]-1, ..., 1, 0 层
        unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ - element_levels_[i] : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
            output.write(linkLists_[i], linkListSize);
    }
}
```

## 关键日志

```plaintext
# maxM0_ 表示第0层元素的最大连接度, maxM_ 是非0层元素的最大连接度
# ef_construction_ 表示元素插入时搜索最近邻居的数目上限; ef_ 类似, 对应在线检索时搜索最近邻居的数目上限
hnsw link list stats: maxM_=20, maxM0_=40, ef_construction_=400, ef_=800
# 各层元素的连接度分布情况, level[i][j] 表示第 i 层中连接度为 j 的候选数目. 注意 maxM0_ = 2 - maxM_. 其中 level 0 日志里 "|" 后表示连接度大于 maxM_ 的节点分布情况
level 0: 0, 0, 0, 0, 0, 0, 170854, 133643, 168370, 212557, 262078, 314615, 402673, 467119, 526643, 581453, 636356, 681868, 723651, 760250, 1316119,  | 1212479, 1195375, 1165219, 1116889, 1063437, 1027709, 975428, 922131, 879651, 836371, 799325, 764842, 725988, 693405, 667227, 640846, 618547, 603457, 599855, 2706684, 
level 1: 0, 0, 0, 0, 0, 0, 27042, 11458, 16140, 22193, 29910, 38712, 48708, 59213, 70483, 83181, 97620, 114334, 135004, 164806, 408413, 
level 2: 0, 0, 0, 0, 0, 0, 561, 474, 907, 1373, 1992, 2697, 3339, 4147, 4751, 5497, 5946, 6695, 7390, 8182, 12902, 
level 3: 0, 0, 0, 0, 0, 0, 8, 36, 66, 103, 141, 198, 268, 287, 298, 336, 309, 338, 313, 311, 323, 
level 4: 0, 0, 0, 0, 0, 0, 9, 14, 26, 14, 13, 11, 13, 15, 13, 8, 6, 4, 4, 8, 1, 
level 5: 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
level 6: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
# hnsw 其实是个多层网络, level[i] 表示第 i 层的元素个数
Nodes Level Info: level[0]: 25245897, level[1]: 1260364, level[2]: 63518, level[3]: 3176, level[4]: 151, level[5]: 7, level[6]: 1, 
# hnsw 包含多少层(一般在1-8层左右, 千万的数据量层数一般是7-8), 总元素个数(sum(level[i]))
HierarchicalNSW::saveIndex cur_element_count = 26573114, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 727, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 440, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 33, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 2, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 5, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 7, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 41, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 92, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 362, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 4, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 19, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 16, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 5, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 6801, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 3, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 860, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 730, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 218, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 327, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 58, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 9215, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 37, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 10, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 6135, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 4813662, maxlevel = 7
HierarchicalNSW::saveIndex cur_element_count = 12836, maxlevel = 4
HierarchicalNSW::saveIndex cur_element_count = 93, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 161, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 687489, maxlevel = 5
HierarchicalNSW::saveIndex cur_element_count = 12, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 1, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 28, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 5, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 68, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 8544, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 6, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 68, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 95, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 2, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 32, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 63, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 77, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 194905826, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16419121, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 15805561, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16170883, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16466969, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16035537, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16181788, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16206240, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 15709760, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16039945, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16390833, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16017525, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 16550154, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 81173502, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 81085517, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 80882302, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 11383722, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 2537458, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 1817924, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 18456300, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 20552119, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 28817711, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 53790513, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 3615804, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 823171, maxlevel = 4
HierarchicalNSW::saveIndex cur_element_count = 7661292, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 82195414, maxlevel = 6
HierarchicalNSW::saveIndex cur_element_count = 3, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 7, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 3494, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 8246, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 113276, maxlevel = 4
HierarchicalNSW::saveIndex cur_element_count = 272, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 2, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 9830, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 4971, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 7874, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 777846, maxlevel = 4
HierarchicalNSW::saveIndex cur_element_count = 3, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 6577, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 182704, maxlevel = 4
HierarchicalNSW::saveIndex cur_element_count = 71917, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 2, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 22, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 6, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 13279, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 14, maxlevel = 1
HierarchicalNSW::saveIndex cur_element_count = 5, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 11, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 5, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 2, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 3, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 486, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 308, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 7881, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 61482, maxlevel = 3
HierarchicalNSW::saveIndex cur_element_count = 7973, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 7595, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 1145, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 3, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 7931, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 7723, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 1, maxlevel = 0
HierarchicalNSW::saveIndex cur_element_count = 761, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 268, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 202, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 5668, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 6151, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 7744, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 5706, maxlevel = 2
HierarchicalNSW::saveIndex cur_element_count = 6193, maxlevel = 2

# 2亿左右数据量
hnsw saveIndex max_elements_ 185460174, cur_element_count: 185460174
hnsw link list stats: maxM_=24, maxM0_=48, ef_construction_=400, ef_=800
level 0: 0, 0, 0, 0, 0, 0, 0, 3712137, 2149431, 2134251, 2248432, 2551366, 2493705, 2504759, 2698100, 2790548, 2890080, 2980148, 3073410, 3158411, 3236171, 3318894, 3391224, 3466871, 9980139,  (邻居个数大于maxM_的节点)| 7157535, 6452754, 6128410, 5791403, 5533491, 5374628, 5220142, 5077278, 4899823, 4676128, 4489081, 4331019, 4209490, 4248810, 4022758, 3860251, 3737898, 3637183, 3554265, 3482896, 3435029, 3472939, 4356206, 19532680, 
level 1: 0, 0, 0, 0, 0, 0, 0, 185367, 15225, 18467, 23804, 30381, 38409, 47932, 59254, 73735, 90687, 112096, 141470, 183627, 246764, 349413, 531663, 894181, 4684595, 
level 2: 0, 0, 0, 0, 0, 0, 0, 6153, 458, 470, 372, 421, 402, 484, 542, 653, 787, 964, 1361, 1995, 3325, 6101, 13352, 30445, 253740, 
level 3: 0, 0, 0, 0, 0, 0, 0, 102, 0, 0, 0, 1, 1, 0, 0, 5, 1, 6, 5, 9, 32, 74, 262, 876, 11991, 
level 4: 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 11, 27, 30, 54, 63, 63, 55, 63, 56, 56, 72, 
level 5: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 
level 6: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
Nodes Level Info: level[0]: 177733104, level[1]: 7405045, level[2]: 308660, level[3]: 12809, level[4]: 538, level[5]: 17, level[6]: 1, 
HierarchicalNSW::saveIndex cur_element_count = 185460174, maxlevel = 6

hnsw saveIndex max_elements_ 185494934, cur_element_count: 185494934
hnsw link list stats: maxM_=24, maxM0_=48, ef_construction_=400, ef_=800
level 0: 0, 0, 0, 0, 0, 0, 0, 4505966, 2219303, 2186491, 2269976, 2375987, 2478425, 2504918, 2701597, 2808617, 2894638, 2984787, 3069500, 3168765, 3237340, 3314012, 3399573, 3464690, 9398218,  (邻居个数大于maxM_的节点)| 6651118, 6219449, 6417555, 5956873, 5588528, 5376902, 5308554, 5097732, 4820635, 4624309, 4511023, 4529831, 4329189, 4218345, 4003958, 3830255, 3714636, 3610976, 3531066, 3468096, 3426839, 3466447, 4344765, 19465050, 
level 1: 0, 0, 0, 0, 0, 0, 0, 235528, 15232, 18594, 23212, 29511, 37615, 46866, 57728, 71678, 88313, 110405, 140546, 181434, 246286, 348617, 530903, 893763, 4652249, 
level 2: 0, 0, 0, 0, 0, 0, 0, 8209, 419, 389, 379, 396, 426, 523, 559, 631, 789, 1092, 1456, 2079, 3319, 6262, 13047, 30485, 251615, 
level 3: 0, 0, 0, 0, 0, 0, 0, 204, 16, 12, 8, 6, 2, 3, 6, 2, 2, 7, 4, 13, 29, 75, 266, 845, 11871, 
level 4: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 13, 19, 30, 46, 53, 64, 58, 63, 70, 65, 68, 
level 5: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 
level 6: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
Nodes Level Info: level[0]: 177766454, level[1]: 7406405, level[2]: 308704, level[3]: 12814, level[4]: 539, level[5]: 17, level[6]: 1, 
HierarchicalNSW::saveIndex cur_element_count = 185494934, maxlevel = 6
```
