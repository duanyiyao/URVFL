class Distribute_MNIST:

    def __init__(self, data_owners, data_loader,device):


        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []

        self.labels = []

        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:
            
            curr_data_dict = {}
            
            # calculate width and height according to the no. of workers for UNIFORM distribution
            height = images.shape[-1]//self.no_of_owner

            self.labels.append(labels)

            # iterate over each worker for distribution of current batch of the self.data_loader
            for i, owner in enumerate(self.data_owners[:-1]):
                
                # split the image and send it to VirtualWorker (which is supposed to be a dataowner or client)
                image_part_ptr = images[:, :, :, height * i : height * (i + 1)]

                curr_data_dict[owner] = image_part_ptr

            # Repeat same for the remaining part of the image
            if(self.no_of_owner == 1):
                i = -1

            last_owner = self.data_owners[-1]
            last_part_ptr = images[:, :, :, height * (i + 1) :]

            curr_data_dict[last_owner] = last_part_ptr

            self.data_pointer.append(curr_data_dict)
            
    def __iter__(self):
        
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (data_ptr, label)
            
    def __len__(self):
        
        return len(self.data_loader)-1
