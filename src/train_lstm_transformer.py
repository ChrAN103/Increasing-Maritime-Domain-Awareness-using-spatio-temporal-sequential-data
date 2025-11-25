import os
# Enable MPS fallback BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


import mlflow
import mlflow.pytorch

import pandas as pd
import torch
from lstm_transformer.lstm_transformer_classifier import LSTMTransformerClassifier
from streamer import BatchStream 
from utils_transformer import SimpleDataset, collate_fn
from torch.utils.data import DataLoader, random_split

def run(hyperparams): 
    best_val_loss = float('inf')
    print("Staring training loop for LSTM-Transformer Classifier...")

    #Setting up for the train loop
    ports_df = pd.read_csv("src/port_counts.csv")  
    port_encoder = {port_name: i for i, port_name in enumerate(ports_df["Port"])} 

    #Loading all data 
    print("Loading data...")
    batchstreamer = BatchStream(batch_size = hyperparams["batch_size"], port_encoder = port_encoder) 
    hyperparams["output_size"] = len(port_encoder) 
    # Get the number of features from the first trajectory tensor (should be 5)
    first_df = batchstreamer.item_to_df(batchstreamer.items[0])
    first_trajectory = first_df.iloc[0]["input_segment"]
    hyperparams["input_size"] = len(first_trajectory[0]) if len(first_trajectory) > 0 else 5 

    print("Streaming data...")
    batchstreamer.stream_data() 

    print("Creating dataset and dataloaders...")
    dataset = SimpleDataset(batchstreamer.batch_X_list, batchstreamer.batch_Y_list)
    train_size = int(hyperparams["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
    train_dataset,
    batch_size=hyperparams["batch_size"],
    shuffle=True, 
    collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )


    print("Initializing model...")
    model = LSTMTransformerClassifier(
        input_size=hyperparams["input_size"],
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_lstm_layers"],
        num_heads=hyperparams["num_heads"],
        num_transformer_layers=hyperparams["num_transformer_layers"],
        output_size=hyperparams["output_size"],
        batch_first=hyperparams["batch_first"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # use Adam optimizer
    optimiser = torch.optim.AdamW(model.parameters(), hyperparams["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]) 

    for epoch in range(hyperparams["num_epochs"]):  
        epoch_metrics = {"epoch": epoch+1} 

        model.train()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0


        for batch_X, mask, lengths, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)  
            batch_Y = batch_Y.to(device)

            # Forward pass with mask
            optimiser.zero_grad()
            outputs = model(batch_X, mask=mask)
            loss = loss_fn(outputs, batch_Y)

            # Backward pass and optimization
            loss.backward()
            optimiser.step()


            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_Y.size(0)
            train_correct += predicted.eq(batch_Y).sum().item()

        #calc epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        epoch_metrics["train_loss"] = avg_train_loss
        epoch_metrics["train_accuracy"] = train_accuracy 
        
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%") 

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0 

        with torch.no_grad():
            for batch_X, mask, lengths, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                mask = mask.to(device)
                lengths = lengths.to(device)  
                batch_Y = batch_Y.to(device)

                outputs = model(batch_X, mask=mask)
                loss = loss_fn(outputs, batch_Y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_Y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / len(val_loader.dataset)
        epoch_metrics["val_loss"] = avg_val_loss
        epoch_metrics["val_accuracy"] = val_accuracy

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        metrics_df = pd.concat([metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True) 
        metrics_df.to_csv("lstm_transformer_training_metrics.csv", index=False) 

                # Log metrics to MLflow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model state locally
            torch.save(model.state_dict(), "best_model_state.pt")
            print(f"New best model (val_loss: {avg_val_loss:.4f})")

    # After training loop ends, log only the best model once
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model and log it to MLflow (only once!)
    model.load_state_dict(torch.load("best_model_state.pt"))
    mlflow.pytorch.log_model(model, "best_model") 

if __name__ == "__main__": 
    hyperparams = {
    "hidden_size": 32*2, 
    "num_lstm_layers": 2, 
    "num_heads": 8, 
    "num_transformer_layers": 2, 
    "batch_size" : 16, 
    "batch_first": True, 
    "learning_rate": 0.001, 
    "train_split": 0.7, 
    "num_epochs": 30, 
    "dropout": 0.2
    }

    mlflow.set_experiment("LSTM-Transformer-Classifier-Experiment - Full Run 1: With seconds as time")
    with mlflow.start_run():
        mlflow.log_params(hyperparams)
        run(hyperparams=hyperparams)