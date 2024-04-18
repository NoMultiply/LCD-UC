
import torch

from data_loader import Dataset


class MFModel(torch.nn.Module):
    def __init__(self, dataset: Dataset, n_hidden, device):
        super(MFModel, self).__init__()
        self.device = device

        self.user_features = torch.FloatTensor(dataset.user_features).to(device)
        self.item_features = torch.FloatTensor(dataset.item_features).to(device)
        self.mlp_user = torch.nn.Sequential(
            torch.nn.Linear(dataset.n_user_features, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(n_hidden, n_hidden),
        )
        self.mlp_item = torch.nn.Sequential(
            torch.nn.Linear(dataset.n_item_features, n_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(n_hidden, n_hidden),
        )

        self.n_output = n_hidden
        self.to(device)

    def get_embeddings(self, users, items):
        user_embeddings = torch.nn.functional.normalize(self.mlp_user(self.user_features[users]), dim=1)
        item_embeddings = torch.nn.functional.normalize(self.mlp_item(self.item_features[items]), dim=1)
        return user_embeddings, item_embeddings

    def get_score(self, user_embeddings, item_embeddings):
        interactions = user_embeddings * item_embeddings
        return torch.sigmoid(torch.sum(interactions, dim=1)), user_embeddings, item_embeddings

    def get_score_for_valid(self, user_embeddings, item_embeddings, users, items):
        return self.get_score(user_embeddings[users], item_embeddings[items])[0]

    def forward(self, users, items):
        user_embeddings, item_embeddings = self.get_embeddings(users, items)
        return self.get_score(user_embeddings, item_embeddings)


class BoxModel(torch.nn.Module):
    def __init__(self, base_model, n_hidden, attn, mask, tau, beta, gd, device):
        super(BoxModel, self).__init__()
        self.base_model = base_model
        self.device = device
        self.attn = attn
        self.mask = mask
        self.tau = tau
        self.beta = beta
        self.gd = gd
        self.n_input = self.base_model.n_output

        self.mlp_uc = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, n_hidden),
        )
        self.mlp_uo = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, n_hidden),
            torch.nn.Softplus(),
        )
        self.mlp_ic = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, n_hidden),
        )
        self.mlp_io = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, n_hidden),
            torch.nn.Softplus(),
        )

        mlps = [self.mlp_uc, self.mlp_uo, self.mlp_ic, self.mlp_io]

        if attn:
            self.mlp_attn = torch.nn.Sequential(
                torch.nn.Linear(self.n_input, n_hidden),
            )
            mlps.append(self.mlp_attn)

        if mask:
            self.mlp_mask = torch.nn.Sequential(
                torch.nn.Linear(self.n_input, n_hidden),
            )
            mlps.append(self.mlp_mask)
        
        for mlp in mlps:
            for child in mlp:
                if hasattr(child, 'weight'):
                    torch.nn.init.xavier_normal_(child.weight)

        self.to(device)

    def get_embeddings(self, users, items):
        user_embeddings, item_embeddings = self.base_model.get_embeddings(
            users, items)
        user_centers = self.mlp_uc(user_embeddings)
        user_offsets = self.mlp_uo(user_embeddings)

        item_centers = self.mlp_ic(item_embeddings)
        item_offsets = self.mlp_io(item_embeddings)
        return (user_embeddings, user_centers, user_offsets), (item_embeddings, item_centers, item_offsets)

    def get_score(self, user_embeddings, item_embeddings):
        user_embeddings, user_centers, user_offsets = user_embeddings
        item_embeddings, item_centers, item_offsets = item_embeddings

        user_uppers = user_centers + user_offsets
        user_lowers = user_centers - user_offsets
        item_uppers = item_centers + item_offsets
        item_lowers = item_centers - item_offsets

        user_uppers = torch.sigmoid(user_uppers)
        user_lowers = torch.sigmoid(user_lowers)
        item_uppers = torch.sigmoid(item_uppers)
        item_lowers = torch.sigmoid(item_lowers)

        if self.gd:
            # Using Gumbel Distribution
            euler_constant = 0.57721566490153286060
            uppers = self.beta * \
                torch.log(torch.exp(user_uppers / self.beta) +
                          torch.exp(item_uppers / self.beta))
            lowers = -self.beta * \
                torch.log(torch.exp(-user_lowers / self.beta) +
                          torch.exp(-item_lowers / self.beta))
            interactions = self.beta * \
                torch.log(1 + torch.exp((uppers - lowers) /
                          self.beta - 2 * euler_constant))
        else:
            # Naive Box Embedding
            interactions = torch.minimum(
                user_uppers, item_uppers) - torch.maximum(user_lowers, item_lowers)

        interactions = -torch.log(torch.sigmoid(-interactions))  # softplus

        if self.attn:
            attn = self.mlp_attn(torch.tanh(user_embeddings * interactions))
            interactions = attn * interactions

        if self.mask:
            mask_weight = self.mlp_mask(user_embeddings)

            delta = torch.rand_like(mask_weight).to(self.device)
            masks = torch.sigmoid(
                (torch.log(delta) - torch.log(1 - delta) + mask_weight) / self.tau)
            
            interactions = interactions * masks

        return (
            torch.sum(interactions, dim=1),
            user_offsets, item_offsets, masks if self.mask else None,
            attn if self.attn else None,
            user_lowers, user_uppers,
            item_lowers, item_uppers,
            user_centers, item_centers
        )

    def get_score_for_valid(self, user_embeddings, item_embeddings, users, items):
        return self.get_score(
            ([x[users] for x in user_embeddings]),
            ([x[items] for x in item_embeddings])
        )[0]

    def forward(self, users, items):
        user_embeddings, item_embeddings = self.get_embeddings(users, items)
        return self.get_score(user_embeddings, item_embeddings)
