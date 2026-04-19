from torch_geometric.utils import k_hop_subgraph
import torch

def get_neighbor_subgraph(full_edge_index, batch_indices, num_hops=1):
    """
        Mở rộng các nút trong batch để bao gồm các nút lân cận k-hop và tạo đồ thị cục bộ.
    
        Args:
             full_edge_index (Tensor): Toàn bộ ma trận cạnh của đồ thị [2, E].
             batch_indices (Tensor/List): Chỉ số của các nút trong mini-batch hiện tại.
            num_hops (int): Số bước mở rộng lân cận (thường là 1 hoặc 2).
   
        Returns:
            subset (Tensor): Danh sách TẤT CẢ các nút (batch gốc + lân cận). Dùng để load features/sequences.
            local_edge_index (Tensor): Ma trận cạnh chỉ chứa các kết nối trong subset, đã được đánh số lại (relabel) từ 0.
            mapping (Tensor): Vị trí của các nút batch gốc trong subset. Dùng để lấy logits sau khi qua GNN.
    """
        # Đảm bảo batch_indices là tensor long
    if not isinstance(batch_indices, torch.Tensor):
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
   
        # Sử dụng hàm k_hop_subgraph của PyG
        # relabel_nodes=True: Rất quan trọng! Nó đánh số lại các nút trong subgraph từ 0 đến N_subgraph - 1
        # để các lớp GNN (GAT/GCN) có thể tính toán được.
    subset, local_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=batch_indices,
        num_hops=num_hops,
        edge_index=full_edge_index,
        relabel_nodes=True
    )
   
    return subset, local_edge_index, mapping