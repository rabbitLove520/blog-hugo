---
title: wxWidgets使用
author: pigLoveRabbit
tags:
  - C++
categories:
  - C++
  - wxWidgets
date: 2022-08-15 09:30:00
---
![](https://www.wxwidgets.org/assets/img/header-logo.png)

<!-- more -->

## 源码编译安装
前往wxWidgets[官网](https://www.wxwidgets.org/downloads/)，下载wxWidgets

![upload successful](/images/wxWidgets_home.png)

### 编译VC版本
打开`Developer Command Prompt for VS 2022`工具，进入 `build\msw` 目录，编译
```
 nmake /f makefile.vc BUILD=debug SHARED=0 TARGET_CPU=X64
```
这里我编译了64位debug版的静态库，这里`x64`和`debug`两个关键字很重要，你在VS中开发时，也要选择相应的配置  
![tip](https://user-images.githubusercontent.com/16663435/175751439-7e86aa58-4ba4-4b8f-8082-25fb3a7d0070.png "tip")  
### 编译gcc的版本
如果你的编译工具链是MinGW-w64，那也是一样进入 `build\msw` 目录，编译
```
mingw32-make -f makefile.gcc SHARED=0 UNICODE=1 BUILD=debug -j8
```

参数说明：
* SHARED=0：编译静态库（SHARED=1 为动态库）
* UNICODE=1：启用 Unicode 支持（必须与应用程序一致）
* BUILD=release：发布版（debug 为调试版）
* -j8：使用 8 个线程并行编译，加快速度

### 编译输出位置
| 编译器 | 静态库输出目录 | 动态库输出目录 | 配置子目录示例 |
|--------|----------------|----------------|----------------|
| **MSVC** | `wxWidgets\lib\vc_lib` | `wxWidgets\lib\vc_dll` | `vc_lib\mswu`（Release Unicode）<br>`vc_lib\mswud`（Debug Unicode） |
| **GCC (MinGW)** | `wxWidgets\lib\gcc_lib` | `wxWidgets\lib\gcc_dll` | `gcc_lib\mswu`（Release Unicode）<br>`gcc_lib\mswud`（Debug Unicode） |

MSVC 编译的库不能与 GCC 编译的应用程序链接，反之亦然（ABI 不兼容）。
> ✅ **关键规律**：`vc_*` 对应 MSVC，`gcc_*` 对应 MinGW；`_lib` 为静态库，`_dll` 为动态库。


## VS中关联操作
在**属性管理器**窗口添加**wxWidgets**目录下的wxwidgets.props文件  
![tip](https://user-images.githubusercontent.com/16663435/175751577-a88e9bbb-c3a2-4321-a93b-3c2ae26be0b6.png "tip")  
这里我们还需要额外注意一下，因为我们跑的是GUI程序，所以需要在项目的**属性**中，把项目设置成`窗口`
![upload](/images/wxWidgets_vs_settings.jpg)  
跑个简单的程序，hello.cpp
```
// wxWidgets "Hello world" Program
// For compilers that support precompilation, includes "wx/wx.h".
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif
class MyApp : public wxApp
{
public:
    virtual bool OnInit();
};
class MyFrame : public wxFrame
{
public:
    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
private:
    void OnHello(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    wxDECLARE_EVENT_TABLE();
};
enum
{
    ID_Hello = 1
};
wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
EVT_MENU(ID_Hello, MyFrame::OnHello)
EVT_MENU(wxID_EXIT, MyFrame::OnExit)
EVT_MENU(wxID_ABOUT, MyFrame::OnAbout)
wxEND_EVENT_TABLE()
wxIMPLEMENT_APP(MyApp);
bool MyApp::OnInit()
{
    MyFrame* frame = new MyFrame("Hello World", wxPoint(50, 50), wxSize(450, 340));
    frame->Show(true);
    return true;
}
MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
    : wxFrame(NULL, wxID_ANY, title, pos, size)
{
    wxMenu* menuFile = new wxMenu;
    menuFile->Append(ID_Hello, "&Hello...\tCtrl-H",
        "Help string shown in status bar for this menu item");
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);
    wxMenu* menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);
    wxMenuBar* menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");
    SetMenuBar(menuBar);
    CreateStatusBar();
    SetStatusText("Welcome to wxWidgets!");
}
void MyFrame::OnExit(wxCommandEvent& event)
{
    Close(true);
}
void MyFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("This is a wxWidgets' Hello world sample",
        "About Hello World", wxOK | wxICON_INFORMATION);
}
void MyFrame::OnHello(wxCommandEvent& event)
{
    wxLogMessage("Hello world from wxWidgets!");
}
```
