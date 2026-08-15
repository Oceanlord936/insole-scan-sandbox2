package com.example.insolescanplan2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.example.insolescanplan2.docscan.DocScanScreen
import com.example.insolescanplan2.ui.theme.InsoleScanplan2Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            InsoleScanplan2Theme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    DocScanScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}