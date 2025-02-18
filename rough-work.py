        ##############  For Regularization ############
        # Fix regularization parameters initialization
        self.max_components = net_params.get('max_components', 10)
        self.reg_lambda = net_params.get('reg_lambda', 0.01)  # Increased from 0.001
        self.reg_target = 1.0
        self.reg_enabled = net_params.get('regularization', {}).get('enabled', True)
        
        # Initialize component weights
        init_type = net_params.get('component_init', 'random')
        if init_type == 'random':
            init_tensor = torch.rand(self.max_components) * 0.5 + 0.5  # Initialize between 0.5 and 1.0
        else:
            init_tensor = torch.ones(self.max_components)
            
        self.component_weights = nn.Parameter(
            init_tensor.to(dtype=torch.float32, device=self.device),
            requires_grad=True
        )
        
        print(f"Regularization config:")
        print(f"- Enabled: {self.reg_enabled}")
        print(f"- Lambda: {self.reg_lambda}")
        print(f"- Target: {self.reg_target}")
        print(f"- Max components: {self.max_components}")
        print(f"- Initial weights: {self.component_weights.data}")


def label_dist_regularizer(self, subgraph_components):
        """
        Compute regularization: sum((ni*wi - 1)^2) for each component i in each subgraph
        Args:
            subgraph_components: List of component lists for each subgraph
        """
        if not subgraph_components:
            return torch.tensor(0.0, device=self.device)
            
        reg_loss = torch.tensor(0.0, device=self.device)
        component_terms = []
        
        for components in subgraph_components:
            for i, component in enumerate(components):
                if i >= self.max_components:
                    break  # Skip if more components than weights
                    
                ni = float(len(component))  # Number of nodes in component
                wi = self.component_weights[i]  # Weight for this component
                print(f"Component {i}: {ni} nodes, weight {wi:.4f}")
                
                # Stronger regularization for larger deviations
                term = ((ni * wi - self.reg_target) ** 2)
                component_terms.append(term)
        
        if component_terms:
            # Stack terms for better gradient flow
            reg_loss = torch.stack(component_terms).mean()
            
        return reg_loss

def loss(self, pred, label, subgraph_components=None):
        # Base classification loss
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        criterion = nn.CrossEntropyLoss(weight=weight)
        base_loss = criterion(pred, label)
        
        # Add regularization term with detailed gradient tracking
        if self.reg_enabled and subgraph_components:
            reg_loss = self.label_dist_regularizer(subgraph_components)
            
            # Dynamic lambda based on losses
            dynamic_lambda = self.reg_lambda * (base_loss.detach() / (reg_loss.detach() + 1e-8))
            total_loss = base_loss + dynamic_lambda * reg_loss
            
            # Gradient debugging
            with torch.no_grad():
                print("\n=== Detailed Loss Analysis ===")
                print(f"Base Loss: {base_loss.item():.4f}")
                print(f"Reg Loss: {reg_loss.item():.4f}")
                print(f"Dynamic Lambda: {dynamic_lambda.item():.4f}")
                print(f"Effective Reg Loss: {(dynamic_lambda * reg_loss).item():.4f}")
            
            # Detailed gradient debugging
            print("\n=== Loss and Gradient Info ===")
            print(f"Base loss: {base_loss.item():.4f}")
            print(f"Reg loss (raw): {reg_loss.item():.4f}")
            print(f"Component weights before backward:")
            print(f"- Values: {self.component_weights.data}")
            print(f"- Requires grad: {self.component_weights.requires_grad}")
            print(f"- Grad fn: {total_loss.grad_fn}")
            
            # Force retain graph for debugging
            total_loss.backward(retain_graph=True)
            
            if self.component_weights.grad is not None:
                print("\nGradient information after backward:")
                print(f"- Gradient norm: {self.component_weights.grad.norm():.4f}")
                print(f"- Gradient values: {self.component_weights.grad}")
                print(f"- Max gradient: {self.component_weights.grad.max():.4f}")
                print(f"- Min gradient: {self.component_weights.grad.min():.4f}")
            else:
                print("\nWARNING: No gradients computed!")
            
            return total_loss
        
        return base_loss