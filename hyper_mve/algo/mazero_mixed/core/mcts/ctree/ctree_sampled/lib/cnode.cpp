#include "cnode.h"

#include <cmath>
#include <stack>
#include <sys/time.h>
#include <map>
#include <cstdlib>
#include <cstring>
#include <omp.h>

namespace tree
{

    CNode::CNode(int agent_num, float prior, float pred_prob, float beta, float beta_hat, bool is_root, float rho, float lam)
        : visit_count(0), num_children(0), hidden_state_index_x(-1), agent_num(agent_num),
          prior(prior), pred_prob(pred_prob), beta(beta), beta_hat(beta_hat),
          is_root(is_root),
          reward_vec(agent_num, 0.f), pred_value_vec(agent_num, 0.f),
          subtree_info(agent_num, tools::SubTreeValueSet(rho, lam)),
          children(), children_action()
    {
        /*
        Overview:
            Initialization of CNode with prior value and root flag. Rewards,
            predicted values and OS(lambda) subtree statistics are per-agent
            vectors of length `agent_num`.
        */
    }

    CNode::~CNode() {}

    bool CNode::expanded()
    {
        return this->num_children > 0;
    }

    float CNode::reward_mean()
    {
        float s = 0.f;
        for (int i = 0; i < this->agent_num; ++i)
            s += this->reward_vec[i];
        return s / this->agent_num;
    }

    float CNode::pred_value_mean()
    {
        float s = 0.f;
        for (int i = 0; i < this->agent_num; ++i)
            s += this->pred_value_vec[i];
        return s / this->agent_num;
    }

    float CNode::value(int agent)
    {
        /*
        Overview:
            Return the average mcts value of the current node for one agent.
        */
        if (! this->expanded())
        {
            return 0;
        }
        else
        {
            return this->subtree_info[agent].value_estimation();
        }
    }

    float CNode::value()
    {
        /*
        Overview:
            Agent-mean mcts value (original team-scalar semantics).
        */
        if (! this->expanded())
        {
            return 0;
        }
        float s = 0.f;
        for (int i = 0; i < this->agent_num; ++i)
            s += this->subtree_info[i].value_estimation();
        return s / this->agent_num;
    }

    float CNode::get_qsa(int agent, float discount)
    {
        return this->reward_vec[agent] + discount * this->value(agent);
    }

    float CNode::get_qsa(float discount)
    {
        return this->reward_mean() + discount * this->value();
    }

    void CNode::get_marginal_visit_count(tools::Array2D<int> logits)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            CNode *child = this->children[i];
            for (size_t j = 0; j < logits.d1; ++j)
            {
                logits(j, this->children_action[i][j]) += child->visit_count;
            }
        }
    }

    void CNode::get_marginal_priors(tools::Array2D<float> priors)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            CNode *child = this->children[i];
            for (size_t j = 0; j < priors.d1; ++j)
            {
                priors(j, this->children_action[i][j]) += child->prior;
            }
        }
    }

    void CNode::get_sampled_visit_count(int *logits)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            logits[i] = this->children[i]->visit_count;
        }
    }

    void CNode::get_sampled_pred_probs(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->pred_prob;
        }
    }

    void CNode::get_sampled_beta(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->beta;
        }
    }

    void CNode::get_sampled_beta_hat(float *probs)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            probs[i] = this->children[i]->beta_hat;
        }
    }

    void CNode::get_sampled_priors(float *priors)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            priors[i] = this->children[i]->prior;
        }
    }

    void CNode::get_sampled_imp_ratio(float *imp_ratio)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            imp_ratio[i] = this->children[i]->beta_hat / this->children[i]->beta * this->children[i]->pred_prob;
        }
    }

    void CNode::get_sampled_pred_values(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->pred_value_mean();
        }
    }

    void CNode::get_sampled_mcts_values(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->value();
        }
    }

    void CNode::get_sampled_rewards(float *rewards)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            rewards[i] = this->children[i]->reward_mean();
        }
    }

    void CNode::get_sampled_qvalues(float *values, float discount)
    {
        for (int i = 0; i < this->num_children; ++i)
        {
            values[i] = this->children[i]->get_qsa(discount);
        }
    }

    void CNode::get_sampled_pred_values_vec(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
            for (int j = 0; j < this->agent_num; ++j)
                values[i * this->agent_num + j] = this->children[i]->pred_value_vec[j];
    }

    void CNode::get_sampled_mcts_values_vec(float *values)
    {
        for (int i = 0; i < this->num_children; ++i)
            for (int j = 0; j < this->agent_num; ++j)
                values[i * this->agent_num + j] = this->children[i]->value(j);
    }

    void CNode::get_sampled_rewards_vec(float *rewards)
    {
        for (int i = 0; i < this->num_children; ++i)
            for (int j = 0; j < this->agent_num; ++j)
                rewards[i * this->agent_num + j] = this->children[i]->reward_vec[j];
    }

    void CNode::get_sampled_qvalues_vec(float *values, float discount)
    {
        for (int i = 0; i < this->num_children; ++i)
            for (int j = 0; j < this->agent_num; ++j)
                values[i * this->agent_num + j] = this->children[i]->get_qsa(j, discount);
    }

    //*********************************************************

    SearchResult::SearchResult(int length_max)
    {
        search_path = (CNode **)malloc(sizeof(CNode *) * (length_max + 2));
    }
    SearchResult::~SearchResult()
    {
        free(search_path);
    }

    //*********************************************************

    CTree::CTree(int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, CNode *node_pool_ptr, unsigned int seed, float rho, float lam, int select_mode)
        : gen(seed), agent_num(agent_num), action_space_size(action_space_size), sampled_times(sampled_times), tot_nodes(0),
          select_mode(select_mode),
          rho(rho), lam(lam),
          node_pool_ptr(node_pool_ptr), root(node_pool_ptr),
          minmax_mean(tree_value_stat_delta_lb),
          minmax_agent(agent_num, tools::CMinMaxStats(tree_value_stat_delta_lb)),
          result(simulation_num)
    {
        /*
        Overview:
            The initialization of CTree. `select_mode` = 0 keeps the original
            joint team-UCB selection over sampled children (exact scalar
            baseline behavior when all agents share the reward); 1 enables
            decoupled per-agent UCB selection over marginal statistics.
        */
    }

    CTree::~CTree()
    {
        for (int i = 0; i < this->tot_nodes; ++i)
        {
            this->node_pool_ptr[i].~CNode();
        }
    }

    void CTree::prepare(const float *reward_vec, const float *value_vec, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises)
    {
        /*
        Overview:
            Expand the root node. reward_vec/value_vec are per-agent (agent_num,).
        */
        new (this->root) CNode(this->agent_num, 1., 1., 1., 1., true, this->rho, this->lam);
        ++(this->tot_nodes);
        this->expand(this->root, 0, reward_vec, value_vec, policy_probs, beta, sampled_times, noise_eps, noises);
        this->root->visit_count += 1;
        for (int i = 0; i < this->agent_num; ++i)
            this->root->subtree_info[i].update(value_vec[i], 0);
    }

    void CTree::expand(CNode *node, int hidden_state_index_x, const float *reward_vec, const float *value_vec, tools::Array2D<float> policy_probs, tools::Array2D<float> beta, int sampled_times, float noise_eps, tools::Array2D<float> noises)
    {
        /*
        Overview:
            Expand the child nodes of the current node. reward_vec/value_vec
            are per-agent vectors (agent_num,).
        */
        node->hidden_state_index_x = hidden_state_index_x;
        for (int i = 0; i < this->agent_num; ++i)
        {
            node->reward_vec[i] = reward_vec[i];
            node->pred_value_vec[i] = value_vec[i];
        }

        // compute beta_hat via beta sampling
        std::map<long, float> beta_hat;
        std::map<long, std::vector<int>> action_map;
        std::vector<std::discrete_distribution<>> dists;
        dists.reserve(this->agent_num);
        for (int i = 0; i < this->agent_num; ++i)
        {
            dists.push_back(std::discrete_distribution<>(&beta(i, 0), &beta(i + 1, 0)));
        }
        for (int k = 0; k < sampled_times; ++k)
        {
            long key = 0;
            std::vector<int> sampled_action(this->agent_num, 0);
            for (int i = 0; i < this->agent_num; ++i)
            {
                sampled_action[i] = dists[i](this->gen);
                key = key * 23333 + sampled_action[i]; // use a prime constant 23333 to prevent from degeneration
            }
            beta_hat[key] += 1.0;
            action_map[key] = sampled_action;
        }

        node->num_children = beta_hat.size();
        node->children.reserve(node->num_children);
        node->children_action.reserve(node->num_children);
        // add children
        for (auto it : beta_hat)
        {
            long key = it.first;
            float count = it.second;
            std::vector<int> sampled_action = action_map[key];
            float betahat_prob = count / sampled_times;
            float beta_prob = 1.0;
            float pred_prob = 1.0;
            float prior = 1.0;
            for (int i = 0; i < this->agent_num; ++i)
            {
                beta_prob *= beta(i, sampled_action[i]);
                pred_prob *= policy_probs(i, sampled_action[i]);
                if(noise_eps > 0){
                    float p = policy_probs(i, sampled_action[i]) * (1-noise_eps) + noises(i, sampled_action[i]) * noise_eps;
                    // p ** (1/tau)  \propto  beta
                    prior *= p;
                }else{
                    prior *= policy_probs(i, sampled_action[i]);
                }
            }
            prior = prior * betahat_prob / beta_prob;
            new (this->node_pool_ptr + this->tot_nodes) CNode(this->agent_num, prior, pred_prob, beta_prob, betahat_prob, false, this->rho, this->lam);
            ++(this->tot_nodes);
            node->children.push_back(this->node_pool_ptr + this->tot_nodes - 1);
            node->children_action.push_back(sampled_action);
        }
    }

    float CTree::ucb_score(CNode *child, float parent_q, int total_children_visit_counts, float pb_c_base, float pb_c_init, float discount)
    {
        /*
        Overview:
            Compute the joint (agent-mean) ucb score of the child — original
            team-scalar semantics, normalized by the mean-q minmax stream.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0)
        {
            value_score = 0;
        }
        else
        {
            value_score = child->get_qsa(discount) - parent_q;
        }

        value_score = this->minmax_mean.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    int CTree::select_child(CNode *node, float pb_c_base, float pb_c_init, float discount, float parent_q)
    {
        /*
        Overview:
            Joint-mode selection (original): argmax of the team ucb over the
            sampled children.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        for (int child_index = 0; child_index < node->num_children; ++child_index)
        {
            CNode *child = node->children[child_index];
            float temp_score = ucb_score(child, parent_q, node->visit_count - 1, pb_c_base, pb_c_init, discount);

            if (max_score < temp_score)
            {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(child_index);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(child_index);
            }
        }

        int child_index = 0;
        if (max_index_lst.size() > 0)
        {
            auto rand_index = this->gen() % max_index_lst.size();
            child_index = max_index_lst[rand_index];
        }
        return child_index;
    }

    int CTree::select_child_decoupled(CNode *node, float pb_c_base, float pb_c_init, float discount)
    {
        /*
        Overview:
            Decoupled per-agent selection: each agent maintains marginal
            statistics of its own actions over the sampled children and
            independently maximizes its own UCB_i (normalized by its own
            minmax stream). The joint tuple of per-agent choices is then
            mapped onto the sampled children: exact match if present,
            otherwise projection to the child with maximum agreement
            (ties broken uniformly at random).
        */
        const float epsilon = 0.000001;
        int total = node->visit_count - 1;
        float pb_c_common = log((total + pb_c_base + 1) / pb_c_base) + pb_c_init;

        std::vector<int> choice(this->agent_num, -1);
        std::vector<int> vis(this->action_space_size);
        std::vector<float> wq(this->action_space_size), pri(this->action_space_size);
        std::vector<char> present(this->action_space_size);

        for (int i = 0; i < this->agent_num; ++i)
        {
            std::fill(vis.begin(), vis.end(), 0);
            std::fill(wq.begin(), wq.end(), 0.f);
            std::fill(pri.begin(), pri.end(), 0.f);
            std::fill(present.begin(), present.end(), 0);
            for (int c = 0; c < node->num_children; ++c)
            {
                CNode *child = node->children[c];
                int a = node->children_action[c][i];
                present[a] = 1;
                pri[a] += child->prior;
                if (child->visit_count > 0)
                {
                    vis[a] += child->visit_count;
                    wq[a] += child->visit_count * child->get_qsa(i, discount);
                }
            }

            float best = FLOAT_MIN;
            std::vector<int> best_lst;
            for (int a = 0; a < this->action_space_size; ++a)
            {
                if (!present[a])
                    continue;
                float pb_c = pb_c_common * (sqrt((float)total) / (vis[a] + 1));
                float prior_score = pb_c * pri[a];
                float value_score = 0.f;
                if (vis[a] > 0)
                {
                    value_score = wq[a] / vis[a] - node->pred_value_vec[i];
                }
                value_score = this->minmax_agent[i].normalize(value_score);
                if (value_score < 0)
                    value_score = 0;
                if (value_score > 1)
                    value_score = 1;
                float s = prior_score + value_score;
                if (best < s)
                {
                    best = s;
                    best_lst.clear();
                    best_lst.push_back(a);
                }
                else if (s >= best - epsilon)
                {
                    best_lst.push_back(a);
                }
            }
            choice[i] = best_lst[this->gen() % best_lst.size()];
        }

        // map the joint tuple of per-agent choices onto the sampled children
        int best_agree = -1;
        std::vector<int> cand;
        for (int c = 0; c < node->num_children; ++c)
        {
            int agree = 0;
            for (int i = 0; i < this->agent_num; ++i)
                agree += (node->children_action[c][i] == choice[i]);
            if (agree > best_agree)
            {
                best_agree = agree;
                cand.clear();
                cand.push_back(c);
            }
            else if (agree == best_agree)
            {
                cand.push_back(c);
            }
        }
        return cand[this->gen() % cand.size()];
    }

    void CTree::select_path(float pb_c_base, float pb_c_init, float discount)
    {
        /*
        Overview:
            Search node path from the root node and store in this->result.
        */
        CNode *node = this->root;
        this->result.search_len = 0;
        this->result.search_path[this->result.search_len] = node;

        while (node->expanded())
        {
            int child_index;
            if(node->is_root && node->visit_count <= node->num_children){
                child_index = node->visit_count - 1;
            }else if(this->select_mode == 1){
                child_index = select_child_decoupled(node, pb_c_base, pb_c_init, discount);
            }else{
                child_index = select_child(node, pb_c_base, pb_c_init, discount, node->pred_value_mean());
            }
            // next
            this->result.action = &(node->children_action[child_index][0]);
            node = node->children[child_index];
            this->result.search_len += 1;
            this->result.search_path[this->result.search_len] = node;
        }

        CNode *parent = this->result.search_path[this->result.search_len - 1];
        this->result.idx = (parent->hidden_state_index_x);
        this->result.leaf = node;
    }

    void CTree::back_propagate(const float *value_vec, float discount)
    {
        /*
        Overview:
            Update per-agent value statistics and visit counts of nodes along
            the search path (vector backup). Both the agent-mean minmax stream
            (joint mode) and the per-agent minmax streams (decoupled mode) are
            maintained.
        */
        std::vector<float> bootstrap(value_vec, value_vec + this->agent_num);
        int path_len = this->result.search_len;

        for (int i = path_len; i >= 0; --i)
        {
            CNode *node = this->result.search_path[i];

            if (i != path_len && i != 0)
            {
                // not leaf, not root: remove the stale q entries
                CNode* father = this->result.search_path[i - 1];
                this->minmax_mean.remove(node->get_qsa(discount) - father->pred_value_mean());
                for (int j = 0; j < this->agent_num; ++j)
                    this->minmax_agent[j].remove(node->get_qsa(j, discount) - father->pred_value_vec[j]);
            }

            node->visit_count += 1;
            for (int j = 0; j < this->agent_num; ++j)
                node->subtree_info[j].update(bootstrap[j], path_len - i);

            if (i != 0)
            {
                // not root: insert the refreshed q entries
                CNode* father = this->result.search_path[i - 1];
                this->minmax_mean.insert(node->get_qsa(discount) - father->pred_value_mean());
                for (int j = 0; j < this->agent_num; ++j)
                    this->minmax_agent[j].insert(node->get_qsa(j, discount) - father->pred_value_vec[j]);
            }

            for (int j = 0; j < this->agent_num; ++j)
                bootstrap[j] = node->reward_vec[j] + discount * bootstrap[j];
        }
    }

    void CTree::expand_and_backprop(int hidden_state_index_x, float discount, int sampled_times, const float *reward_vec, const float *value_vec, tools::Array2D<float> policy_prob, tools::Array2D<float> beta)
    {
        tools::Array2D<float> _(nullptr, 0, 0);   // placeholder
        expand(this->result.leaf, hidden_state_index_x, reward_vec, value_vec, policy_prob, beta, sampled_times, 0., _);
        back_propagate(value_vec, discount);
    }

    void CTree::get_root_value(float *val)
    {
        *val = this->root->value();
    }
    void CTree::get_root_value_vec(float *val)
    {
        for (int j = 0; j < this->agent_num; ++j)
            val[j] = this->root->value(j);
    }
    void CTree::get_root_marginal_visit_count(tools::Array2D<int> logits)
    {
        root->get_marginal_visit_count(logits);
    }
    void CTree::get_root_marginal_priors(tools::Array2D<float> priors)
    {
        root->get_marginal_priors(priors);
    }
    void CTree::get_root_sampled_actions(tools::Array2D<int> actions)
    {
        tools::my_assert(root->children_action.size() == actions.d1 && root->children_action[0].size() == actions.d2,
                         "Error in `CTree::get_root_sampled_actions`: dimensions of `root->children_action` does not match that of the receiving buffer.");
        for (size_t i = 0; i < actions.d1; ++i)
            for (size_t j = 0; j < actions.d2; ++j)
                actions(i, j) = root->children_action[i][j];
    }
    void CTree::get_root_sampled_visit_count(int *logits)
    {
        root->get_sampled_visit_count(logits);
    }
    void CTree::get_root_sampled_pred_probs(float *probs)
    {
        root->get_sampled_pred_probs(probs);
    }
    void CTree::get_root_sampled_imp_ratio(float *imp_ratio)
    {
        root->get_sampled_imp_ratio(imp_ratio);
    }
    void CTree::get_root_sampled_beta(float *probs)
    {
        root->get_sampled_beta(probs);
    }
    void CTree::get_root_sampled_beta_hat(float *probs)
    {
        root->get_sampled_beta_hat(probs);
    }
    void CTree::get_root_sampled_priors(float *priors)
    {
        root->get_sampled_priors(priors);
    }
    void CTree::get_root_sampled_pred_values(float *values)
    {
        root->get_sampled_pred_values(values);
    }
    void CTree::get_root_sampled_mcts_values(float *values)
    {
        root->get_sampled_mcts_values(values);
    }
    void CTree::get_root_sampled_rewards(float *rewards)
    {
        root->get_sampled_rewards(rewards);
    }
    void CTree::get_root_sampled_qvalues(float *values, float discount)
    {
        root->get_sampled_qvalues(values, discount);
    }
    void CTree::get_root_sampled_pred_values_vec(float *values)
    {
        root->get_sampled_pred_values_vec(values);
    }
    void CTree::get_root_sampled_mcts_values_vec(float *values)
    {
        root->get_sampled_mcts_values_vec(values);
    }
    void CTree::get_root_sampled_rewards_vec(float *rewards)
    {
        root->get_sampled_rewards_vec(rewards);
    }
    void CTree::get_root_sampled_qvalues_vec(float *values, float discount)
    {
        root->get_sampled_qvalues_vec(values, discount);
    }

    void CTree::print()
    {
        for (int i = 0; i < this->tot_nodes; ++i)
        {
            fprintf(stderr, "node %d info:\n", i);
            auto u = this->node_pool_ptr[i];
            fprintf(stderr, "\tvisit count: %d, idx: %d\n", u.visit_count, u.hidden_state_index_x);
            fprintf(stderr, "\treward(mean): %f, prior: %f, estimate_value(mean): %f\n", u.reward_mean(), u.prior, u.value());
            fprintf(stderr, "\tchildren (%d in total):\n", u.num_children);
            for (int j = 0; j < u.num_children; ++j)
            {
                fprintf(stderr, "\t\tid: %ld,\t action: ", u.children[j] - this->node_pool_ptr);
                for (auto it : u.children_action[j])
                    fprintf(stderr, "%d ", it);
                fprintf(stderr, "\n");
            }
        }
    }

    //*********************************************************

    CTree_batch::CTree_batch(int root_num, int agent_num, int action_space_size, int sampled_times, int simulation_num, float tree_value_stat_delta_lb, unsigned int random_seed, float rho, float lam, int select_mode)
    {
        /*
        Overview:
            The initialization of CTree_batch.
        */
        this->root_num = root_num;
        this->agent_num = agent_num;
        this->action_space_size = action_space_size;
        this->pool_size_per_root = sampled_times * (simulation_num + 2);
        this->thread_num = 1;

        // allocate memory
        this->node_pool = (CNode *)malloc(sizeof(CNode) * this->root_num * this->pool_size_per_root);
        this->trees = (CTree *)malloc(sizeof(CTree) * this->root_num);
        tools::my_assert(this->node_pool && this->trees, "Error in `CTree_batch::CTree_batch`: `malloc` fails for `node_pool` or `trees`.");

        // init each tree
        for (int i = 0; i < this->root_num; ++i)
        {
            auto ptr_i = this->node_pool + i * this->pool_size_per_root;
            unsigned int seed_i = random_seed * 2333 + i;
            new (this->trees + i) CTree(agent_num, action_space_size, sampled_times, simulation_num, tree_value_stat_delta_lb, ptr_i, seed_i, rho, lam, select_mode);
        }
    }

    CTree_batch::~CTree_batch()
    {
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].~CTree();
        }
        free(this->trees);
        free(this->node_pool);
    }

    void CTree_batch::prepare(float *rewards_buf, float *values_buf, float *policy_probs_buf, float *beta_buf, int sampled_times, float noise_eps, float* noises_buf)
    {
        /*
        Overview:
            Expand the root nodes of a batch.
        Arguments:
            - rewards_buf: batch root node per-agent rewards (root_num, agent_num).
            - values_buf: batch root node per-agent values (root_num, agent_num).
            - policy_probs_buf: batch root node probs (root_num, agent_num, action_space_size)
            - beta_buf: batch root node sampling-dists (root_num, agent_num, action_space_size)
        */
        tools::Array3D<float> policy_probs(policy_probs_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> beta(beta_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> noises(noises_buf, this->root_num, this->agent_num, this->action_space_size);

        for (int i = 0; i < this->root_num; ++i)
        {
            tools::Array2D<float> prob_i(&policy_probs(i, 0, 0), policy_probs.d2, policy_probs.d3);
            tools::Array2D<float> beta_i(&beta(i, 0, 0), beta.d2, beta.d3);
            tools::Array2D<float> noise_i(&noises(i, 0, 0), noises.d2, noises.d3);
            this->trees[i].prepare(rewards_buf + i * this->agent_num, values_buf + i * this->agent_num, prob_i, beta_i, sampled_times, noise_eps, noise_i);
        }
    }

    void CTree_batch::cbatch_selection(float pb_c_base, float pb_c_init, float discount, int *idx_buf, int *idy_buf, int *act_buf)
    {
        int *idx_arr = idx_buf;
        int *idy_arr = idy_buf;
        tools::Array2D<int> act_arr(act_buf, this->root_num, this->agent_num);

        #pragma omp parallel for num_threads(this->thread_num)
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].select_path(pb_c_base, pb_c_init, discount);
            idx_arr[i] = this->trees[i].result.idx;
            idy_arr[i] = i;
            for (int j = 0; j < this->agent_num; ++j)
                act_arr(i, j) = this->trees[i].result.action[j];
        }
    }

    void CTree_batch::cbatch_expansion_and_backup(int hidden_state_index_x, float discount, int sampled_times, float *rewards_buf, float *values_buf, float *policy_probs_buf, float *beta_buf)
    {
        /*
        Overview:
            Expand and backup the leaf nodes of a batch.
        Arguments:
            - rewards_buf: batch leaf node per-agent rewards (root_num, agent_num).
            - values_buf: batch leaf node per-agent values (root_num, agent_num).
        */
        tools::Array3D<float> policy_probs(policy_probs_buf, this->root_num, this->agent_num, this->action_space_size);
        tools::Array3D<float> beta(beta_buf, this->root_num, this->agent_num, this->action_space_size);

        #pragma omp parallel for num_threads(this->thread_num)
        for (int i = 0; i < this->root_num; ++i)
        {
            tools::Array2D<float> prob_i(&policy_probs(i, 0, 0), policy_probs.d2, policy_probs.d3);
            tools::Array2D<float> beta_i(&beta(i, 0, 0), beta.d2, beta.d3);
            this->trees[i].expand_and_backprop(hidden_state_index_x, discount, sampled_times, rewards_buf + i * this->agent_num, values_buf + i * this->agent_num, prob_i, beta_i);
        }
    }

    void CTree_batch::get_roots_values(float *buf)
    {
        float *val = buf;
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_value(val + i);
        }
    }

    void CTree_batch::get_roots_values_vec(float *buf)
    {
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_value_vec(buf + i * this->agent_num);
        }
    }


    void CTree_batch::get_roots_marginal_visit_count(int *buf)
    {
        memset(buf, 0, sizeof(int) * this->root_num * this->agent_num * this->action_space_size);
        tools::Array3D<int> arr(buf, this->root_num, this->agent_num, this->action_space_size);
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_marginal_visit_count(tools::Array2D<int>(&arr(i, 0, 0), this->agent_num, this->action_space_size));
        }
    }

    void CTree_batch::get_roots_marginal_priors(float *buf)
    {
        memset(buf, 0, sizeof(float) * this->root_num * this->agent_num * this->action_space_size);
        tools::Array3D<float> arr(buf, this->root_num, this->agent_num, this->action_space_size);
        for (int i = 0; i < this->root_num; ++i)
        {
            this->trees[i].get_root_marginal_priors(tools::Array2D<float>(&arr(i, 0, 0), this->agent_num, this->action_space_size));
        }
    }

    int CTree_batch::get_num_children_of_root(int tree_id)
    {
        return this->trees[tree_id].root->num_children;
    }

    void CTree_batch::get_root_sampled_actions(int tree_id, int *buf)
    {
        tools::Array2D<int> arr(buf, this->trees[tree_id].root->num_children, this->agent_num);
        this->trees[tree_id].get_root_sampled_actions(arr);
    }

    void CTree_batch::get_root_sampled_visit_count(int tree_id, int *buf)
    {
        memset(buf, 0, sizeof(int) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_visit_count(buf);
    }

    void CTree_batch::get_root_sampled_pred_probs(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_pred_probs(buf);
    }

    void CTree_batch::get_root_sampled_imp_ratio(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_imp_ratio(buf);
    }

    void CTree_batch::get_root_sampled_beta(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_beta(buf);
    }

    void CTree_batch::get_root_sampled_beta_hat(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_beta_hat(buf);
    }

    void CTree_batch::get_root_sampled_priors(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_priors(buf);
    }

    void CTree_batch::get_root_sampled_rewards(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_rewards(buf);
    }

    void CTree_batch::get_root_sampled_pred_values(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_pred_values(buf);
    }

    void CTree_batch::get_root_sampled_mcts_values(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_mcts_values(buf);
    }

    void CTree_batch::get_root_sampled_qvalues(int tree_id, float *buf, float discount)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children);
        this->trees[tree_id].get_root_sampled_qvalues(buf, discount);
    }

    void CTree_batch::get_root_sampled_pred_values_vec(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children * this->agent_num);
        this->trees[tree_id].get_root_sampled_pred_values_vec(buf);
    }

    void CTree_batch::get_root_sampled_mcts_values_vec(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children * this->agent_num);
        this->trees[tree_id].get_root_sampled_mcts_values_vec(buf);
    }

    void CTree_batch::get_root_sampled_rewards_vec(int tree_id, float *buf)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children * this->agent_num);
        this->trees[tree_id].get_root_sampled_rewards_vec(buf);
    }

    void CTree_batch::get_root_sampled_qvalues_vec(int tree_id, float *buf, float discount)
    {
        memset(buf, 0, sizeof(float) * this->trees[tree_id].root->num_children * this->agent_num);
        this->trees[tree_id].get_root_sampled_qvalues_vec(buf, discount);
    }

    void CTree_batch::print()
    {
        for (int i = 0; i < root_num; ++i)
        {
            fprintf(stderr, "---------- Tree %d info ----------\n", i);
            this->trees[i].print();
            fprintf(stderr, "\n");
        }
    }
}

// ===============================================================
